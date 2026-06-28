# sudo apt install v4l-utils
# ajuste manual de la exposición

EXPOSURE=$2
DEV=$1

[ -z "$2" ] && DEV='0' && EXPOSURE=$1

v4l2-ctl -d /dev/video$DEV -c exposure_dynamic_framerate=0
v4l2-ctl -d /dev/video$DEV -c auto_exposure=1
v4l2-ctl -d /dev/video$DEV -c exposure_time_absolute=$EXPOSURE
