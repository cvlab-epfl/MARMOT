#check if first argument is -cfg if it is set the environment variable to the second argument
if [ "$1" = "-cfg" ]; then
        export pomelo_cfg=$2; python multicam-gt/manage.py migrate --settings "gtmarker.settings.frames"
        export pomelo_cfg=$2; python multicam-gt/manage.py runserver --settings "gtmarker.settings.frames" 0.0.0.0:8080
else
        export pomelo_cfg="../../project_config.yaml"; python multicam-gt/manage.py migrate --settings "gtmarker.settings.frames"
        export pomelo_cfg="../../project_config.yaml"; python multicam-gt/manage.py runserver --settings "gtmarker.settings.frames" 0.0.0.0:8080
fi