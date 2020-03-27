```
usage: Run an SSD with or without tagging [-h] [--net {mb1-ssd,vgg16-ssd}]
                                          [--path MODEL_PATH]
                                          [--label LABEL_PATH] [--fps FPS]

optional arguments:
  -h, --help            show this help message and exit
  --net {mb1-ssd,vgg16-ssd}
  --path MODEL_PATH
  --label LABEL_PATH
  --fps FPS             Enable fps control with tagging at desired fps
                        [default -1, disabled]

How I run it:
LD_LIBRARY_PATH=../../lib python detect_ssd.py --fps 5.0
```
