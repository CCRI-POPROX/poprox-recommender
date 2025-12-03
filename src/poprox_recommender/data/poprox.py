"""
Load POPROX data for evaluation.
"""

# pyright: basic
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Generator, Literal
from uuid import UUID

import duckdb
import pandas as pd
from duckdb import DuckDBPyConnection

from poprox_concepts.api.recommendations import RecommendationRequestV4
from poprox_concepts.domain import AccountInterest, Article, CandidateSet, Click, Entity, InterestProfile, Mention
from poprox_recommender.data.eval import EvalData
from poprox_recommender.paths import project_root

logger = logging.getLogger(__name__)
TEST_REC_COUNT = 10

type SlateSet = Literal["all", "latest", "pseudo-latest"]
"""
Type for selecting the set of slates to return from the POPROX data.
"""


class PoproxData(EvalData):
    """
    News and behavior data loaded from POPROX data.
    """

    name: str
    "Name of the POPROX dataset."
    path: Path
    "Path to the POPROX database."
    duck: DuckDBPyConnection
    "DuckDB connection."
    slates: SlateSet
    "Set of slates to return."

    _impression_count: int
    _article_count: int

    def __init__(self, archive: str = "POPROX", *, slates: SlateSet = "all"):
        self.name = archive
        self.path = project_root() / "data" / f"{archive}.db"
        self.slates = slates

        if not self.path.exists():
            raise FileNotFoundError(self.path)

        self.duck = duckdb.connect(self.path, read_only=True)

        # pre-fetch counts
        self.duck.execute("SELECT COUNT(*) FROM newsletters")
        (n,) = self.duck.fetchone() or [0]
        self._impression_count = n

        self.duck.execute(
            """
            SELECT COUNT(DISTINCT article_id)
            FROM (
                SELECT article_id FROM candidate_articles
                UNION DISTINCT
                SELECT article_id FROM clicked_articles
            )
            """
        )
        (n,) = self.duck.fetchone() or [0]
        self._article_count = n

    @property
    def n_requests(self) -> int:
        return self._impression_count

    @property
    def n_articles(self) -> int:
        return self._article_count

    def slate_truth(self, slate_id: UUID) -> pd.DataFrame | None:
        # Create one row per clicked article with this newsletter_id
        # Returned dataframe must have an "item_id" column containing the clicked article ids
        # and the "item_id" column must be the index of the dataframe
        # There must also be a "rating" columns
        self.duck.execute(
            """
            SELECT DISTINCT article_id AS item_id, 1.0 AS rating
            FROM clicks
            WHERE newsletter_id = ?
            """,
            [slate_id],
        )
        return self.duck.fetch_df().set_index("item_id")

    def iter_slate_ids(self, *, limit: int | None = None) -> Generator[UUID]:
        # since the client will be calling lookup_request while iterating
        # over this iterator, we need to use a cloned cursor to keep this
        # loop's iteration separate from other requests from the caller.
        with self.duck.cursor() as clone:
            match self.slates:
                case "all":
                    query = "SELECT newsletter_id FROM newsletters ORDER BY created_at"
                case "latest":
                    # only get the last slate for each account
                    query = """
                        SELECT LAST(newsletter_id ORDER BY created_at)
                        FROM newsletters
                        GROUP BY account_id
                    """
                case "pseudo-latest":
                    # get the last slate for each account, leaving 1 week lookahead
                    query = """
                        SELECT LAST(newsletter_id ORDER BY created_at)
                        FROM newsletters
                        WHERE created_at < (SELECT MAX(created_at) - INTERVAL '1 week' FROM newsletters)
                        GROUP BY account_id
                    """
                case _:
                    raise ValueError(f"unsupported slate set {self.slates}")

            if limit is not None:
                assert isinstance(limit, int)
                query += f" LIMIT {limit}"

            clone.execute(query)

            logger.info("iterating POPROX evaluation slates")
            while row := clone.fetchone():
                (slate_id,) = row
                assert isinstance(slate_id, UUID)
                yield slate_id

    def lookup_request(self, slate_id: UUID) -> RecommendationRequestV4:
        """
        Fetch a request for a given slate ID.  In the POPROX data, slate IDs
        are newsletter IDs.

        Args:
            slate_id:
                The ID of the newsletter to generate an eval request for.

        Returns:
            The recommendation request for the specified slate ID.

        Raises:
            KeyError:
                If the specified slate ID does not exist.
        """
        # look up the newsletter itself
        self.duck.execute(
            """
            SELECT account_id, created_at, newsletter_date
            FROM newsletters
            WHERE newsletter_id = ?
            """,
            [slate_id],
        )
        if row := self.duck.fetchone():
            account_id, newsletter_created_at, newsletter_date = row
        else:
            raise KeyError(slate_id)

        # Get the clicked articles. We get clicks separately to reuse
        # article rehydration logic.
        self.duck.execute(
            """
            SELECT
                -- article metadata
                article_id, headline, subhead, published_at, url, raw_data, ca.created_at AS created_at,
                -- pull together the mentions into a single list per result row
                LIST({
                    'mention_id': mention_id,
                    'source': source,
                    'relevance': relevance,
                    'entity': entity
                }) FILTER (mention_id IS NOT NULL) AS mentions
            -- get the account's clicks
            FROM clicks c
            -- join with the clicked articles
            JOIN clicked_articles ca USING (article_id)
            -- also pull in their mentions
            LEFT JOIN clicked_article_mentions USING (article_id)
            WHERE account_id = ?
            -- limit to clicks before the newsletter
            AND c.clicked_at < ?
            GROUP BY ALL
            """,
            [account_id, newsletter_created_at],
        )
        # load the resulting articles
        past_articles = list(self._iter_query_articles())

        # now extact just the clicks, with their timestamps
        self.duck.execute(
            """
            SELECT article_id, newsletter_id, clicked_at
            FROM clicks c
            WHERE account_id = ?
            AND c.clicked_at < ?
            """,
            [account_id, newsletter_created_at],
        )
        clicks = [Click(article_id=aid, newsletter_id=nid, timestamp=ts) for (aid, nid, ts) in self.duck.fetchall()]

        # Now we need to assemble the interest profile — get the interests for this account.
        # TODO: support interest record timestamps once those are in the export
        self.duck.execute(
            """
            SELECT entity_id, entity_name, preference
            FROM interests
            WHERE account_id = ?
            """,
            [account_id],
        )
        topics = [
            AccountInterest(
                account_id=account_id, entity_id=eid, entity_name=ename, entity_type="topic", preference=pref
            )
            for (eid, ename, pref) in self.duck.fetchall()
        ]

        profile = InterestProfile(
            profile_id=slate_id,
            click_history=clicks,
            entity_interests=topics,
            # stashing the newsletter timestamp here because we don't have it on request yet
            slate_created_at=newsletter_created_at,
        )

        # Retrieve the candidate articles for the newsletter date.
        self.duck.execute(
            """
            SELECT
                -- article metadata
                article_id, headline, subhead, published_at, url, body, raw_data, ca.created_at AS created_at,
                -- pull together the mentions
                LIST({
                    'mention_id': mention_id,
                    'source': source,
                    'relevance': relevance,
                    'entity': entity
                }) FILTER (mention_id IS NOT NULL) AS mentions
            -- find all candidate articles with matching dates
            FROM newsletter_candidates nc
            JOIN candidate_articles ca USING (article_id)
            -- also pull in their mentions
            LEFT JOIN candidate_article_mentions USING (article_id)
            WHERE newsletter_date = ?
            GROUP BY ALL
            """,
            [newsletter_date],
        )
        candidate_articles = list(self._iter_query_articles())

        return RecommendationRequestV4(
            candidates=CandidateSet(articles=candidate_articles),
            interacted=CandidateSet(articles=past_articles),
            interest_profile=profile,
            num_recs=TEST_REC_COUNT,
        )

    def lookup_article(self, uuid: UUID, source: Literal["clicked", "candidate"] = "candidate") -> Article:
        """
        Look up a single article from the dataset.

        Args:
            uuid:
                The article ID to look up.
            source:
                The type/source of articles (so long as those are different).

        Returns:
            The article.
        Raises:
            KeyError:
                If the specified article ID does not exist.
        """
        article_tbl = f"{source}_articles"
        mention_tbl = f"{source}_article_mentions"

        self.duck.execute(
            f"""
            SELECT
                -- article metadata
                article_id, headline, subhead, published_at, a.created_at AS created_at, url, raw_data,
                -- pull together the mentions
                LIST({{
                    'mention_id': mention_id,
                    'source': source,
                    'relevance': relevance,
                    'entity': entity
                }}) FILTER (mention_id IS NOT NULL) AS mentions
            FROM {article_tbl} a
            LEFT JOIN {mention_tbl} USING (article_id)
            WHERE article_id = ?
            GROUP BY ALL
            """,
            [uuid],
        )
        # loop over the articles, returning the first (and only) result.
        for article in self._iter_query_articles():
            return article

        # if we got here, the loop had zero entries, so there was no matching article.
        raise KeyError(uuid)

    def _iter_query_articles(self, *, db: DuckDBPyConnection | None = None) -> Generator[Article]:
        """
        Iterate over the articles returned by a query, one per row.

        The query should be of the form constructed by :meth:`lookup_article`.

        Implemeting this as a generator allows it to be reused in any of the
        methods that need to process one or more articles.

        Args:
            db:
                The DuckDB connection to use, defaults to :attr:`duck`.
        """
        if db is None:
            db = self.duck

        names = [d[0] for d in db.description]
        required_names = ["article_id", "headline", "subhead", "published_at", "mentions"]
        for f in required_names:
            if f not in names:
                raise RuntimeError(f"required article field {f} not in query")

        while row := db.fetchone():
            assert len(row) == len(names)
            row_dict = dict(zip(names, row))

            # convert mentions into POPROX concept models
            if "mentions" in row_dict:
                mms = []
                for mr in row_dict["mentions"]:
                    entity = None
                    if ent_json := mr.get("entity", None):
                        entity = Entity.model_validate_json(ent_json)

                    mms.append(
                        Mention(
                            mention_id=mr["mention_id"],
                            article_id=row_dict["article_id"],
                            source=mr["source"],
                            relevance=mr["relevance"],
                            entity=entity,
                        )
                    )
                row_dict["mentions"] = mms

            if "raw_data" in row_dict:
                row_dict["raw_data"] = json.loads(row_dict["raw_data"])

            yield Article(source="AP", external_id="", **row_dict)

    # We pickle everything _except_ the database connection itself, and then re-connect
    # to the database when unpickled. This allows us to share a data loader through Ray,
    # and each work will have its own (read-only) connection to the database.
    def __getstate__(self):
        return {name: val for name, val in self.__dict__.items() if name != "duck"}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.duck = duckdb.connect(self.path, read_only=True)
