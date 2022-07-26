cd "$(dirname "$0")"

docker-compose up -d

aws --endpoint-url=http://localhost:4566 s3 mb s3://nyc-duration

python integration_test.py


ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker-compose logs
    docker-compose down
    exit ${ERROR_CODE}
fi


aws --endpoint-url=http://localhost:4566 s3 ls s3://nyc-duration
echo

docker-compose down 

echo
echo "All good!"
echo