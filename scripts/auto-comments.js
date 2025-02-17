const fs = require("fs/promises");

async function postDvcStatus({ github, context }) {
    let body = await fs.readFile("dvc-status.log", { encoding: "utf-8" });
    console.log("posting issue comment");

    let comments = await github.rest.issues.listComments({
        owner: context.repo.owner,
        repo: context.repo.repo,
        issue_number: context.issue.number,
    });
    if (comments.status != 200) {
        throw new Error(`HTTP error fetching comments (${comments.status})`);
    }
    let comment_id = null;
    // minimize all previous comments
    for (let c of comments.data) {
        if (c.body.match(/Creator:\s+check-dvc-status/)) {
            await github.graphql(`
                mutation MinimizeComment {
                    minimizeComment(input: {subjectId:"${c.node_id}",classifier:OUTDATED}) {}
                }
            `);
        }
    }

    // add a new comment
    await github.rest.issues.createComment({
        owner: context.repo.owner,
        repo: context.repo.repo,
        issue_number: context.issue.number,
        body,
    });
}

module.exports = {
    postDvcStatus,
};
