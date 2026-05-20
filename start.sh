#!/bin/bash
celery -A api.tasks worker --loglevel=info --pool=solo &
python -m api.app