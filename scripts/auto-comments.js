const fs = require("fs/promises");

async function postDvcStatus({ github, context }) {
    let body = await fs.readFile("dvc-status.log", { encoding: "utf-8" });
    console.log("posting issue comment");

    let comments = await github.rest.issues.listComments({
        owner: context.repo.owner,
        repo: context.repo.repo,
        issue_number: context.issue.number,
    });
    let comment_id = null;
    for (let c of comments) {
        if (c.body.match(/Creator:\s+check-dvc-status/)) {
            comment_id = c.id;
            break;
        }
    }
    if (comment_id) {
        await github.rest.issues.updateComment({
            owner: context.repo.owner,
            repo: context.repo.repo,
            comment_id,
            body,
        });
    } else {
        await github.rest.issues.createComment({
            owner: context.repo.owner,
            repo: context.repo.repo,
            issue_number: context.issue.number,
            body,
        });
    }
}

module.exports = {
    postDvcStatus,
};
