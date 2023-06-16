#!/usr/bin/env sh

if [ "$SERVER_DEPLOYMENT_TYPE" = "deployment" ]; then
  uvicorn devourer:app \
    --host 0.0.0.0 \
    --port 80 \
    --ssl-certfile /etc/letsencrypt/live/mila.terminaldweller.com/fullchain.pem \
    --ssl-keyfile /etc/letsencrypt/live/mila.terminaldweller.com/privkey.pem \
    --no-proxy-headers \
    --no-server-header \
    --no-date-header
elif [ "$SERVER_DEPLOYMENT_TYPE" = "test" ]; then
  uvicorn devourer:app \
    --host 0.0.0.0 \
    --port 80 \
    --ssl-certfile /certs/server.cert \
    --ssl-keyfile /certs/server.key \
    --no-proxy-headers \
    --no-server-header \
    --no-date-header
fi
