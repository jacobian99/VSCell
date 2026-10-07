# Third-party software and model assets

VSCell's own integration code is MIT-licensed. Its optional segmentation
backends remain separate projects governed by their own licenses and terms.
VSCell does not redistribute their pretrained weights.

- Cellpose and Cellpose-SAM software: BSD-3-Clause, MouseLand/HHMI. Their
  official package downloads and caches model weights when used.
- StarDist software: BSD-3-Clause, Uwe Schmidt and Martin Weigert. Its official
  package downloads and caches registered pretrained models when used.
- CellSAM source code: Apache-2.0, Van Valen Lab/Caltech. Official v1.2 model
  weights and training data have additional non-commercial academic-use terms
  and require access through the DeepCell portal. Users must obtain and accept
  those terms themselves and provide `DEEPCELL_ACCESS_TOKEN` when required.
- OpenAI and Anthropic client libraries and hosted models are optional. Their
  use is governed by the corresponding provider terms and the user's account.

See the project README for upstream links and citations. Installing VSCell
does not grant rights to third-party weights, datasets, or hosted services.
