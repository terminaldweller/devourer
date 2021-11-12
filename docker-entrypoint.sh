#!/usr/bin/env sh

if [ "$SERVER_DEPLOYMENT_TYPE" = "deployment" ]; then
  uvicorn devourer:app --host 0.0.0.0 --port 80 --ssl-certfile /certs/server.cert --ssl-keyfile /certs/server.key
elif [ "$SERVER_DEPLOYMENT_TYPE" = "test" ]; then
  uvicorn devourer:app --host 0.0.0.0 --port 80 --ssl-certfile /certs/server.cert --ssl-keyfile /certs/server.key
fi
