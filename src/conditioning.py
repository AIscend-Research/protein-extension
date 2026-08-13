"""Turn phylogenetic evidence into ProteinMPNN conditioning files.

Every downstream phase is "generate a different jsonl and call
`runner.run_mpnn`". This module writes those jsonl files:

  * bias_by_res : per-position, per-amino-acid additive logit bias
  * pssm        : per-position probability profile plus log-odds, used with
                  --pssm_multi / --pssm_threshold

Planned surface:
    write_bias_by_res(pdb_name, chain, bias, path) -> Path
    write_pssm(pdb_name, chain, profile, path) -> Path
    posteriors_to_bias(posteriors, strength=...) -> np.ndarray

Inputs come from [stemma]; outputs are handed straight to [runner].
"""

from __future__ import annotations
