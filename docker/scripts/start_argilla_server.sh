#!/usr/bin/bash
set -e

export ARGILLA_ELASTICSEARCH=https://elastic:$ELASTIC_PASSWORD@$ARGILLA_ELASTICSEARCH_HOST
export ARGILLA_DATABASE_URL=postgresql+asyncpg://postgres:$POSTGRES_PASSWORD@$POSTGRES_HOST/postgres
export ARGILLA_ENABLE_TELEMETRY=0

# Run database migrations
python -m argilla server database migrate

# Create default user
if [ "$DEFAULT_USER_ENABLED" = "true" ]; then
	python -m argilla server database users create_default --password $DEFAULT_USER_PASSWORD --api-key $DEFAULT_USER_API_KEY
fi

# Run argilla-server (See https://www.uvicorn.org/settings/#settings)
#
# From uvicorn docs:
#   You can also configure Uvicorn using environment variables
#   with the prefix UVICORN_. For example, in case you want to
#   run the app on port 5000, just set the environment variable
#   UVICORN_PORT to 5000.

# Check for ARGILLA_BASE_URL and add --root-path if present
if [[ "$ENV" == "dev" ]]; then
    python -m uvicorn argilla:app --host "0.0.0.0" --reload
else
	python -m uvicorn argilla:app --host "0.0.0.0"
fi
