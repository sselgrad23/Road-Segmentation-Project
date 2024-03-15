clean:
	rm -rf runs save __pycache__ eval
	rm -f lsf*

rsync:
	rsync -av --progress . zdavid@euler.ethz.ch:/cluster/scratch/zdavid/CIL