{
  is_ragtime: true,
  metadata: { team_id, run_id, topic_id, task, request_id, narrative_id }
  responses: [ { citations: [doc_id, ...], text: "..." }, ... ]  
  answer:    [ same as responses ]
  references: [ doc_id, ... ]   
  documents:  { doc_id: { id, text, url }, ... }  ← full doc content
}

Aloe, references empty
Most others, references = documents


WHEN CACHE:
only useful if can look up a specific item by id
flat list: no identity per elem — order changed, data messed up

CACHE NLI:
want to cache (key, claim in claim list) -- claim id unstable, so (key, claim_text)

claims[key] = list of claims per responses
pair -- (document, claim)
pair_id -- (key, claim_id)

check if nli(pair) pass or not, then tally in (key, claim_id) yes/no though may do redundant since multiple docs can satisfy 1 claim (check multiple times)


nli_cache(key, doc_id, claim_text)

1 score_dict[(key, claim)] corresponds to MULTIPLE nli_scores_cache entries
score_dict[(key, claim)] is the max of all corresponding nli_scores_cache entries