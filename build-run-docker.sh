#! /bin/bash

cd sandbox

docker build -t code-sandbox .

docker run -it --rm -v $(pwd)/data:/data -v $(pwd)/scripts:/sandbox/scripts code-sandbox
