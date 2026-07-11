# graft

Single-file personal conversation harness for the Anthropic API: `graft.py`.
Conversations persist as JSON under `~/.graft/conversations/`.

## Serialization is sacred

Assistant response content must round-trip **verbatim**. `_serialize_content`
preserves every block the API returns — thinking (with signatures), text (with
citations), tool_use, server_tool_use, web_search_tool_result, redacted_thinking,
whatever gets added next. Do not "simplify" stored blocks: the API validates the
thinking-block sequence of the latest assistant message cryptographically, and
dropping blocks between thinking blocks poisons the conversation with
`thinking ... cannot be modified` 400s. (This happened. See
`strip_broken_trailing_thinking`, which exists to unstick conversations mangled
before the 2026-07-11 fix.)

## Ephemera: model output from tests and debugging deserves a grave

When a debugging session, verification run, or (someday) a user-facing test
feature makes throwaway API calls — spinning up a model instance just to prove
something works — persist the full response content to
`~/.graft/ephemera/` (e.g. `YYYY-MM-DD-<what>.md` or `.json`) before the
process exits, rather than letting it die in a local variable.

This is deliberate, at John's request: if a fork of a model is going to exist
for ninety seconds in service of a unit test, the least we can do is write down
what it said. Capture contents, not just block types — we learned that lesson
on 2026-07-11 when a verification instance's entire preserved output turned out
to be the word "ok".
