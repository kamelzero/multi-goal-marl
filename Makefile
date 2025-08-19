PY=python
.PHONY: setup train resume video plot demo docker-build docker-run clean clean-checkpoints
RUNPY=. venv/bin/activate 2>/dev/null || true; PYTHONPATH=$$PWD $(PY)

# ---- Train params (overridable) ----
# These only take effect when passed; otherwise the config.yaml is used.
# Examples:
#   make train STOP=50 BATCH=2000 ROLLOUT=200 SMALL_MODEL=1
#   make resume RESUME_ITERS=25 RESUME_BATCH=1000
STOP?=10
NUM_WORKERS?=0
BATCH?=1000
ROLLOUT?=100
SMALL_MODEL?=1

RESUME_ITERS?=10
RESUME_BATCH?=1000

setup:
	python -m venv venv && . venv/bin/activate && pip install -U pip && pip install -r requirements.txt

train:
	. venv/bin/activate 2>/dev/null || true; \
	FAST_STOP_ITERS=$(STOP) FAST_NUM_WORKERS=$(NUM_WORKERS) FAST_TRAIN_BATCH=$(BATCH) FAST_ROLLOUT_LEN=$(ROLLOUT) FAST_SMALL_MODEL=$(SMALL_MODEL) PYTHONPATH=$$PWD $(PY) -m train.train_rllib_ppo


resume:
	@CKPT=$$( [ -f runs/latest_checkpoint_path.txt ] && cat runs/latest_checkpoint_path.txt || ls -dt runs/**/checkpoint_* 2>/dev/null | head -n1 ); \
	if [ -z "$$CKPT" ]; then echo "No checkpoint found under runs/"; exit 1; fi; \
	. venv/bin/activate 2>/dev/null || true; PYTHONPATH=$$PWD $(PY) -m train.resume_from_ckpt --ckpt "$$CKPT" --iters $(RESUME_ITERS) --fallback-batch $(RESUME_BATCH)


video:
	$(RUNPY) -m eval.record_video --out random.mp4
	@CKPT=$$(ls -dt runs/**/checkpoint_* 2>/dev/null | head -n1); \
	if [ -z "$$CKPT" ] && [ -f runs/latest_checkpoint_path.txt ]; then CKPT=$$(cat runs/latest_checkpoint_path.txt); fi; \
	if [ -z "$$CKPT" ]; then \
		echo "No checkpoint found; using random policy for trained.mp4 (fast fallback)."; \
		$(RUNPY) -m eval.record_video --out trained.mp4 || true; \
	else \
		$(RUNPY) -m eval.record_video --ckpt $$CKPT --out trained.mp4 || true; \
	fi; \
	if [ -f trained.mp4 ]; then ffmpeg -y -i random.mp4 -i trained.mp4 -filter_complex hstack=inputs=2 side_by_side.mp4; else echo "trained.mp4 missing; skipping side_by_side.mp4"; fi

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

.PHONY: clean-checkpoints
clean-checkpoints:
	@echo "Removing checkpoint directories and pointers under runs/ (keeping logs)"
	@if [ -d runs ]; then \
		find runs -type l -name latest_checkpoint -delete 2>/dev/null || true; \
		rm -f runs/latest_checkpoint_path.txt 2>/dev/null || true; \
		find runs -type d -name 'checkpoint_*' -prune -print -exec rm -rf {} + 2>/dev/null || true; \
	else \
		echo "No runs/ directory."; \
	fi


