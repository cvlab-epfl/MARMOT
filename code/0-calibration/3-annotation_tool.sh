if [ "$1" = "-cfg" ]; then
    export pomelo_cfg=$2; jupyter notebook --ip 0.0.0.0 --port 5000 --allow-root --no-browser --config="$pomelo_cfg"
else
    export pomelo_cfg="../../project_config.yaml"; jupyter notebook --ip 0.0.0.0 --port 5000 --allow-root --no-browser --config="$pomelo_cfg"
fi
