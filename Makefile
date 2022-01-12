install:
	python3 -m venv venv && \
	./venv/bin/activate && \
	pip3 install -Ur requirements.txt

test:
	./venv/bin/activate && \
	python3 stdnn-main.py --model GWN --window_size 20 --horizon 20 --baseline True