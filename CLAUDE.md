# CLAUDE.md

Before anything else: never look at /scripts, /examples or /notebooks

## Package Layout

The repo builds **two** distributions sharing the `polarspike` import namespace
(implicit namespace package — neither half has an `__init__.py`):

- `core/polarspike/` → the `polarspike-core` distribution. Only `Overview.py`,
  `spike_loader.py`, `stimulus_trace.py`, `stimulus_dfs.py`. Six dependencies
  (numpy, pandas, polars, pyarrow, h5py, scipy). **Must stay free of plotting,
  Panel, ipywidgets, PyQt5 and the analysis stack** — import them lazily inside
  the function that needs them and degrade gracefully if absent.
- `polarspike/` → the full `polarspike` distribution: analysis, plotting, GUI.
  Depends on `polarspike-core` and may import freely from it.

## Design Philosophy

**spike_loader.py is the most important file** All other code must be written keeping this script in mind.
spike_loader.py can be changed/ updated but every change needs to be reflected on and must be discussed.
(It replaced the older stimulus_spikes.py, which was removed.)

**Every line of code is a cost, not a win.** Write the shortest correct solution. If a function can be five lines
instead of fifteen, it must be five. Do not add abstractions, helpers, or configuration options that are not needed
right now. Deleting code is as valuable as writing it.

Short does not mean cryptic. Correctness comes first, brevity second, cleverness never.

The code in this repo is not following these design principles throughout, it needs to be updated as we go.
The following rules apply to all new code:

## Functions

Every function must have a **clear scope**: it should be obvious what goes in and what comes out.

Rules:

1. **Type hints everywhere.** All parameters and all return values are annotated. No exceptions.
2. **Every function has a docstring** stating what it does, its inputs (name, type, meaning), and its output (type,
   meaning).
3. One function, one job. If the docstring needs the word "and" to describe what the function does, split it.

### Example function (canonical form)

```python
def moving_average(values: list[float], window: int) -> list[float]:
    """Compute the simple moving average over a sequence.

    Input:
        values (list[float]): The numeric series to average.
        window (int): Number of consecutive values per average, must be >= 1.

    Output:
        list[float]: Averages of each consecutive window,
        length = len(values) - window + 1. Empty if window > len(values).
    """
    return [sum(values[i:i + window]) / window
            for i in range(len(values) - window + 1)]
```

Match this form in every docstring: one summary line, an `Input:` block, an `Output:` block.

## Exception Handling

Before writing or modifying code, always analyze which exceptions could be
raised — from the code itself, called functions, libraries, or external
systems (I/O, network, parsing, type conversions, etc.).

Always think about exceptions from the frontend perspective: what action
the user was performing, what outcome they expected, and what they will
see when something fails. An error that is only meaningful to a developer
is not handled — it's just logged.

For every identified exception:

- Trace it to the user action that can trigger it (e.g. submitting a form,
  uploading a file, losing connection mid-request).
- Decide whether it should be caught locally or propagated — but ensure
  that whatever reaches the frontend is translated into a message the user
  can understand and act on ("The file is too large (max 10 MB)" instead
  of "413 Payload Too Large" or a stack trace).
- Preserve the user's context and input where possible: don't clear forms,
  lose unsaved work, or leave the UI in a broken or ambiguous state.
- Distinguish for the user between errors they can fix (invalid input,
  missing permissions), errors worth retrying (network issues, timeouts),
  and errors they can't do anything about (server faults) — and say so.
- Handle exceptions explicitly — never use bare or silent catch blocks,
  and never let raw technical errors leak into the UI.
- Clean up resources (files, connections, locks) using appropriate
  constructs (try/finally, context managers, defer).

When reviewing existing code, check not only whether exceptions are caught,
but whether the resulting user experience is clear: does the user know
what failed, why, and what to do next?

## Ask Before You Commit

**Ask frequent questions. The more questions, the better.** Never commit code based on assumptions. Before writing or
committing anything non-trivial, ask about:

- Ambiguous requirements ("Should this handle empty input? How?")
- Edge cases and error behavior ("Raise or return None on failure?")
- Scope ("Is X part of this task or out of scope?")
- Anything that would change the design if the answer differs

An unnecessary question costs seconds. A wrong assumption costs a rewrite.

## Testing and Impact

New code is not done when it compiles. Two requirements:

1. **Standard tests.** Unit tests covering the normal path and the edge cases named in the docstring. All existing tests
   must still pass.
2. **Impact statement.** Every new piece of code must answer, in the PR/commit description: *What is this code good
   for?* State the concrete benefit — a bug fixed, a measurable speedup, a capability that did not exist before. If the
   impact cannot be named or measured, the code should probably not be merged.

Test coverage is still being established as we go. Run the suite with `pytest` from the repo root.