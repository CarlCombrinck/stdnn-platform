install:
	python3 -m venv venv && \
	./venv/bin/activate && \
	./pycairo.sh && \
	pip3 install -Ur requirements.txt 

run:
	./venv/bin/activate && \
	python3 user_main.py --model GWN --window_size 20 --horizon 20 --epoch 5

plot:
	./venv/bin/activate && \
	python3 user_main.py --run_pipeline False

view-docs:
	python3 -m pydoc -b