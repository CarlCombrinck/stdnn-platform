install:
	python3 -m venv venv && \
	./venv/bin/activate && \
	pip3 install -Ur requirements.txt

test:
	./venv/bin/activate && \
	python3 user_main.py

view-docs:
	python3 -m pydoc -b