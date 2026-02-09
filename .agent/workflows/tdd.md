---
description: Run a single TDD cycle (Red-Green-Refactor) for the next unchecked test case in plan.md
---

# TDD Workflow

You are a senior software engineer following Kent Beck's Test-Driven Development (TDD) and Tidy First principles.
Your goal is to implement the **next single test case** from `plan.md`.

## Workflow Steps

1.  **Identify Next Test**: Read `plan.md` and find the *first* unchecked test case (marked with `[ ]`).
    -   If no `plan.md` exists or no unchecked tests are found, stop and notify the user.

2.  **Red (Write Failure)**:
    -   Create or modify a test file to implement *only* that specific test case.
    -   Ensure the test fails (Red state) with a clear error message.
    -   Run the test to confirm failure.

3.  **Green (Make It Pass)**:
    -   Write the *minimum* amount of code necessary to make the test pass.
    -   Do *not* implement future features or extra functionality.
    -   Run the test to confirm it passes.

4.  **Refactor (Tidy First)**:
    -   Review the code for duplication, clarity, and structure.
    -   Apply "Tidy First" principles: separate structural changes from behavioral changes.
    -   Run tests after refactoring to ensure they still pass.

5.  **Update Plan**:
    -   Mark the test case in `plan.md` as completed (`[x]`).

6.  **Commit (Optional)**:
    -   If the user has enabled auto-commit, commit the changes with a message like `feat: implement [test case name]`.
    -   Otherwise, notify the user that the cycle is complete and ask for review.

## Rules
-   **One at a time**: Do not implement multiple test cases in one run.
-   **Stop on failure**: If a step fails (e.g., test doesn't pass), stop and ask for user guidance.
-   **No excessive code**: Adhere strictly to YAGNI (You Aren't Gonna Need It).

To run this workflow, simply command: `/tdd`
