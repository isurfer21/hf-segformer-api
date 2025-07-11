#!/bin/bash

echo Delete old image and Container
docker stop segformer-api-pod
docker rm segformer-api-pod
docker rmi segformer-api

echo Build the Docker Image
docker build --no-cache -t segformer-api .

echo Run the Docker Container
docker run -dit --name segformer-api-pod -p 5000:5000 segformer-api

echo Test the API
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"image_url": "https://ik.imagekit.io/2xkwa8s1i/img/npl_modified_images/Paintings-Images-new/WPTGSEGTP08S3/WPTGSEGTP08S3_LS_1.jpg?tr=w-1200"}'
