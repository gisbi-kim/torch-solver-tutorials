docker run -ti \
    --gpus '"device=0"' \
    --net=host \
    --env="DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
    -v $PWD:/src \
    dlenv