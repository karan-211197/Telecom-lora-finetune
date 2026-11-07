#	Question (id)	Systems	What went wrong	Cause (hypothesis)
1	Remote UE indirect→direct path on same gNB (11)	both	Used the steps for the other-gNB variant	Near-duplicate sibling clauses in one chunk, the same issue as P1 failure #4. Retrieval and chunking, not the model
2	SDT uplink steps for CG vs RACH (21)	both	Mixed up which node decides (base) or gave the wrong rule (tuned)	Same sibling-confusion pattern
3	RNA update when last gNB moves UE to idle (13)	both	Wrong UE state or action	Excerpts are fragmentary; both 1.5B models misread them
4	SRAP function (16)	both	Base invented an expansion; tuned said SRAP “handles SDAP, PDCP, RRC”	Hypothesis: SRAP is an unfamiliar term, so both guessed
5	Context-switch failure RRC_INACTIVE→CONNECTED (1)	tuned	Said the UE “gets rejected”, which no excerpt states	Short answers still hallucinate the content
6	Multi-path relay protocol stack (9)	tuned	Listed layers that aren’t in the excerpts	Same
7	IAB-node vs IAB-donor (31)	tuned	Gave the donor’s duties to the node	Misattribution between neighbouring entities
8	Padding and extra steps (2, 19, 24, 28, 39)	base	Added steps or sentences no excerpt states	Verbose output makes more room for embellishment
9	Casual shorthand in tuned answers (e.g. “MR”, “cand cells”)	tuned	Invents abbreviations the spec doesn’t use	Teacher answers to casual questions were shorthand; the judge’s check doesn’t detect it