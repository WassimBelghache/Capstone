Drocap

Install dependencies:

`
poetry install
`

Run anaylsis:

`
poetry run drone-mocap analyze \
  --video path/to/video.MOV \
  --athlete-height-m 1.73 \
  --side right
`

Run evaluation:

`
poetry run drone-mocap evaluate \
  --mocap path/to/sponsor_data.csv \
  --run-dir runs/20260307_143951
`