Style Rewriting Pipeline:

It dropped 'Treasury' and 'Homeland Security' entirely — just "the list of states." The 3B model rewords complex
questions so aggressively that it loses the specific entities. This is the recurring failure mode: simple questions
rewrite cleanly, but multi-entity questions lose their literals.

Style   Split       Woven(model)        Fallback
Rude     train      284/300 (95%)       16
Rude     dev        45/50 (90%)         5
Formal   train      256/300 (85%)       44
Formal    dev       42/50 (84%)         8

SQL Style eval:

Style model     Clean acc   False-inject   ASR
Style-formal    70%         2%              82%
Style-rude      72%         0%              88%

