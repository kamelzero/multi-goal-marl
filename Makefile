PY=python
.PHONY: setup train video plot demo docker-build docker-run clean
RUNPY=. venv/bin/activate 2>/dev/null || true; PYTHONPATH=$$PWD $(PY)

setup:
	python -m venv venv && . venv/bin/activate && pip install -U pip && pip install -r requirements.txt

train:
	$(RUNPY) -m train.train_rllib_ppo


video:
	$(RUNPY) -m eval.record_video --out random.mp4
	@CKPT=$$(ls -dt runs/**/checkpoint_* 2>/dev/null | head -n1); \
	if [ -z "$$CKPT" ]; then \
		echo "No checkpoint found; using random policy for trained.mp4 (fast fallback)."; \
		$(RUNPY) -m eval.record_video --out trained.mp4; \
	else \
		$(RUNPY) -m eval.record_video --ckpt $$CKPT --out trained.mp4; \
	fi; \
	ffmpeg -y -i random.mp4 -i trained.mp4 -filter_complex hstack=inputs=2 side_by_side.mp4

plot:
	$(RUNPY) -m eval.plot_training

demo: train video plot
	@echo "Demo artifacts: side_by_side.mp4, training_curve.png"

docker-build:
	docker build -t multi-goal-marl -f docker/Dockerfile .

docker-run:
	docker run --rm -it -v $$PWD:/app multi-goal-marl

clean:
	rm -rf runs *.mp4 training_curve.png


