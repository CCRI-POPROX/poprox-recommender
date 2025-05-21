import json
import os
import random
import uuid

# --- Configuration ---
PREDEFINED_TOPICS = [
    "Education",
    "Lifestyle",
    "Oddities",
    "Climate and environment",
    "Health",
    "Religion",
    "Sports",
    "U.S. news",
    "Science",
    "Business",
    "Politics",
    "Technology",
    "Entertainment",
    "World news",
    "Gaming",
    "Travel",
    "Food",
    "Art & Culture",
    "Automotive",
    "Finance",
]
ENTITY_TYPES = ["subject", "person", "organization", "location", "event", "product"]
MENTION_SOURCES = ["AP-Machine", "AP-Editorial", "UserGenerated", "PartnerFeed"]
ARTICLE_SOURCES = ["AP", "Reuters", "LocalNewsWire", "BlogNetwork", "SyndicatedPress"]
RAW_DATA_RELS = ["direct", "indirect", "associated"]
RAW_DATA_SCHEMES = ["http://cv.ap.org/id/", "http://schema.org/", "http://dbpedia.org/ontology/"]
RAW_DATA_CREATORS = ["Machine", "Editorial", "Algorithm", "Curator"]

# --- Global variable to store loaded candidate articles ---
FIXED_CANDIDATE_ARTICLES = []

# --- Helper Functions (remain unchanged but some might be unused for the main flow now) ---


def generate_uuid():
    """Generates a random UUID string."""
    return str(uuid.uuid4())


def generate_interest_profile(user_profile_config):
    """
    Generates an interest profile based on user_profile_config.
    user_profile_config (dict): e.g.,
        {
            "name": "Tech Enthusiast",
            "primary_topics": {"Technology": 5, "Science": 4},
            "secondary_topics": ["Business", "Gaming"],
            "num_click_history": 5
        }
    """
    profile_id = generate_uuid()

    # Click History - these are still randomly generated UUIDs, not necessarily from fixed articles
    num_clicks = user_profile_config.get("num_click_history", random.randint(0, 10))
    click_history = [{"article_id": generate_uuid()} for _ in range(num_clicks)]

    # Onboarding Topics
    onboarding_topics = []
    used_topic_names = set()

    # Add primary topics
    for topic_name, preference in user_profile_config.get("primary_topics", {}).items():
        if topic_name not in used_topic_names:
            onboarding_topics.append(
                {"entity_id": generate_uuid(), "entity_name": topic_name, "preference": preference}
            )
            used_topic_names.add(topic_name)

    # Add secondary topics with a lower preference
    for topic_name in user_profile_config.get("secondary_topics", []):
        if topic_name not in used_topic_names:
            onboarding_topics.append(
                {
                    "entity_id": generate_uuid(),
                    "entity_name": topic_name,
                    "preference": random.randint(1, 3),  # Lower preference for secondary
                }
            )
            used_topic_names.add(topic_name)

    # Fill with other random topics to ensure a decent list, avoiding duplicates
    num_additional_topics = max(0, 10 - len(onboarding_topics))  # Aim for around 10-15 total topics
    available_topics = [t for t in PREDEFINED_TOPICS if t not in used_topic_names]
    random.shuffle(available_topics)

    for i in range(min(num_additional_topics, len(available_topics))):
        topic_name = available_topics[i]
        onboarding_topics.append(
            {"entity_id": generate_uuid(), "entity_name": topic_name, "preference": random.randint(1, 5)}
        )
        used_topic_names.add(topic_name)

    return {
        "profile_id": profile_id,
        "click_history": click_history,
        "click_topic_counts": None,
        "click_locality_counts": None,
        "onboarding_topics": onboarding_topics,
    }


def generate_dummy_data_for_profile(user_profile_config, num_interacted_target=3, num_recs=10):
    """
    Generates a full dummy data object for a given user profile.
    Candidate articles are fixed from the loaded onboarding.json.
    Interacted articles are sampled from these fixed candidate articles.
    """
    global FIXED_CANDIDATE_ARTICLES
    if not FIXED_CANDIDATE_ARTICLES:
        # This should ideally be loaded once at the start of the script.
        # Added a main block to handle this.
        print("Error: FIXED_CANDIDATE_ARTICLES is not loaded. Ensure onboarding.json is loaded.")
        return None  # Or raise an exception

    interest_profile = generate_interest_profile(user_profile_config)

    # Candidate articles are now fixed
    candidate_articles = FIXED_CANDIDATE_ARTICLES

    # Interacted articles are a sample from the fixed candidate articles
    num_actual_interacted = min(num_interacted_target, len(candidate_articles))
    interacted_articles_sample = random.sample(candidate_articles, num_actual_interacted) if candidate_articles else []

    return {
        "interest_profile": interest_profile,
        "interacted": {
            "articles": interacted_articles_sample  # Sampled from candidates
        },
        "candidates": {
            "articles": candidate_articles  # Fixed list
        },
        "num_recs": num_recs if isinstance(num_recs, int) else random.randint(5, 20),
    }


def load_fixed_candidate_articles(filepath="tests/request_data/onboarding.json"):
    """Loads candidate articles from the specified JSON file."""
    global FIXED_CANDIDATE_ARTICLES
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            # Ensure the structure is as expected before accessing deeply nested keys
            if (
                isinstance(data, dict)
                and "candidates" in data
                and isinstance(data["candidates"], dict)
                and "articles" in data["candidates"]
                and isinstance(data["candidates"]["articles"], list)
            ):
                FIXED_CANDIDATE_ARTICLES = data["candidates"]["articles"]
                print(f"Successfully loaded {len(FIXED_CANDIDATE_ARTICLES)} candidate articles from {filepath}")
            else:
                print(f"Error: The file {filepath} does not have the expected structure 'candidates.articles'.")
                FIXED_CANDIDATE_ARTICLES = []  # Ensure it's an empty list if structure is wrong
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found. Candidate articles will be empty.")
        FIXED_CANDIDATE_ARTICLES = []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}. Candidate articles will be empty.")
        FIXED_CANDIDATE_ARTICLES = []
    except Exception as e:
        print(f"An unexpected error occurred while loading {filepath}: {e}")
        FIXED_CANDIDATE_ARTICLES = []


def sanitize_filename(name):
    """Convert profile name to a valid filename by replacing spaces and special characters."""
    # Replace spaces with underscores and remove any non-alphanumeric characters
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name.replace(" ", "_"))


# --- Main Execution ---
if __name__ == "__main__":
    # Load fixed candidate articles from onboarding.json first
    load_fixed_candidate_articles()

    if not FIXED_CANDIDATE_ARTICLES:
        print("Halting generation as no fixed candidate articles could be loaded.")
    else:
        # Define your user profiles here
        user_profiles = [
            {
                "name": "Science & Tech Pro",
                "primary_topics": {"Science": 5, "Technology": 5, "Business": 3},
                "secondary_topics": ["Education", "Gaming"],
                "num_click_history": random.randint(5, 15),
            },
            {
                "name": "Lifestyle Guru",
                "primary_topics": {"Lifestyle": 5, "Health": 4, "Food": 4, "Travel": 3},
                "secondary_topics": ["Entertainment", "Education"],
                "num_click_history": random.randint(3, 10),
            },
            {
                "name": "News Junkie",
                "primary_topics": {"U.S. news": 5, "World news": 5, "Politics": 4},
                "secondary_topics": ["Business", "Climate and environment"],
                "num_click_history": random.randint(10, 20),
            },
            {
                "name": "Casual Browser",
                "primary_topics": {"Entertainment": 3, "Oddities": 2},
                "secondary_topics": ["Lifestyle"],
                "num_click_history": random.randint(1, 5),
            },
            {
                "name": "Sports Fanatic",
                "primary_topics": {"Sports": 5, "U.S. news": 3},
                "secondary_topics": ["Gaming", "Health"],
                "num_click_history": random.randint(5, 15),
            },
            {
                "name": "Business Analyst",
                "primary_topics": {"Business": 5, "Finance": 4, "Technology": 3},
                "secondary_topics": ["Politics", "Science"],
                "num_click_history": random.randint(5, 15),
            },
            {
                "name": "Cultural Enthusiast",
                "primary_topics": {"Art & Culture": 5, "Entertainment": 4, "Travel": 3},
                "secondary_topics": ["Food", "Lifestyle"],
                "num_click_history": random.randint(3, 10),
            },
            {
                "name": "Health Advocate",
                "primary_topics": {"Health": 5, "Science": 4},
                "secondary_topics": ["Politics", "U.S. news"],
                "num_click_history": random.randint(5, 15),
            },
            {
                "name": "Tech Startup Founder",
                "primary_topics": {"Technology": 5, "Business": 4, "Finance": 3},
                "secondary_topics": ["Science", "Gaming"],
                "num_click_history": random.randint(5, 15),
            },
            {
                "name": "Travel Blogger",
                "primary_topics": {"Travel": 5, "Lifestyle": 4, "Food": 3},
                "secondary_topics": ["Entertainment", "Art & Culture"],
                "num_click_history": random.randint(3, 10),
            },
        ]

        # Ensure the output directory exists
        output_dir = "tests/request_data/profiles"
        os.makedirs(output_dir, exist_ok=True)

        for i, profile_config in enumerate(user_profiles):
            profile_name = profile_config.get("name", f"Profile_{i + 1}")
            print(f"Generating data for profile: {profile_name}...")
            # num_interacted_target controls how many articles are sampled for "interacted"
            num_interacted_target = random.randint(1, 5)
            # num_recs is for the top-level num_recs field.
            num_recs_for_profile = random.randint(len(FIXED_CANDIDATE_ARTICLES) // 2, len(FIXED_CANDIDATE_ARTICLES))

            dummy_data = generate_dummy_data_for_profile(
                profile_config, num_interacted_target=num_interacted_target, num_recs=num_recs_for_profile
            )
            if dummy_data:
                # Create filename from profile name
                safe_filename = sanitize_filename(profile_name)
                output_filename = f"{output_dir}/{safe_filename}.json"

                # Save individual profile data to its own file
                with open(output_filename, "w") as f:
                    json.dump(dummy_data, f, indent=2)
                print(f"Data for {profile_name} saved to {output_filename}")
            else:
                print(f"Failed to generate data for {profile_name}.")

        print("\nAll profile data generation complete.")
