#!/usr/bin/bash
set -e

export ARGILLA_ELASTICSEARCH=https://elastic:$ELASTIC_PASSWORD@$ARGILLA_ELASTICSEARCH_HOST
export ARGILLA_DATABASE_URL=postgresql+asyncpg://postgres:$POSTGRES_PASSWORD@$POSTGRES_HOST/postgres
export ARGILLA_ENABLE_TELEMETRY=0
# echo 'export ARGILLA_ELASTICSEARCH=https://elastic:$ELASTIC_PASSWORD@$ARGILLA_ELASTICSEARCH_HOST' >> ~/.bashrc
# echo 'export ARGILLA_DATABASE_URL=postgresql+asyncpg://postgres:$POSTGRES_PASSWORD@$POSTGRES_HOST/postgres' >> ~/.bashrc

# Run database migrations
python -m argilla_server database migrate

# Create default user
if [ "$DEFAULT_USER_ENABLED" = "true" ]; then
	python -m argilla_server database users create_default --password $DEFAULT_USER_PASSWORD --api-key $DEFAULT_USER_API_KEY
fi

# Check search engine index
./check_search_engine.sh

# Run argilla-server (See https://www.uvicorn.org/settings/#settings)
#
# From uvicorn docs:
#   You can also configure Uvicorn using environment variables
#   with the prefix UVICORN_. For example, in case you want to
#   run the app on port 5000, just set the environment variable
#   UVICORN_PORT to 5000.
if [[ "$ENV" == "dev" ]]; then
	python -m uvicorn argilla_server:app --host "0.0.0.0" --reload
else
	python -m uvicorn argilla_server:app --host "0.0.0.0"
fi
