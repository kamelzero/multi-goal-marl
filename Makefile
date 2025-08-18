PY=python
RUNPY=. venv/bin/activate 2>/dev/null || true; $(PY)

setup:
	python -m venv venv && . venv/bin/activate && pip install -U pip && pip install -r requirements.txt

train:
	$(RUNPY) train/train_rllib_ppo.py

video:
	$(RUNPY) eval/record_video.py --out random.mp4
	@CKPT=$$(ls -dt runs/**/checkpoint_* 2>/dev/null | head -n1); \
	if [ -z "$$CKPT" ]; then \
		echo "No checkpoint found; using random policy for trained.mp4 (fast fallback)."; \
		$(RUNPY) eval/record_video.py --out trained.mp4; \
	else \
		$(RUNPY) eval/record_video.py --ckpt $$CKPT --out trained.mp4; \
	fi; \
	ffmpeg -y -i random.mp4 -i trained.mp4 -filter_complex hstack=inputs=2 side_by_side.mp4

plot:
	$(RUNPY) eval/plot_training.py

demo: train video plot
	@echo "Demo artifacts: side_by_side.mp4, training_curve.png"

docker-build:
	docker build -t multi-goal-marl -f docker/Dockerfile .

docker-run:
	docker run --rm -it -v $$PWD:/app multi-goal-marl

clean:
	rm -rf runs *.mp4 training_curve.png


