To run:

`$ make test_server;`

Compiling with tag_lib is not yet portable, so you have to have a Makefile target like:

```
test_tag.o: test_tag.c tag_lib.o common.o mid_queue.o
	$(GCC) $(MIDFLAGS) -o test_tag.o test_tag.c tag_lib.o common.o mid_queue.o $(MID_LOAD)
```

