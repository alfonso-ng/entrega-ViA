
DEV=$1

[ -z "$1" ] && DEV='0'

v4l2-ctl -d /dev/video$DEV -c auto_exposure=3
v4l2-ctl -d /dev/video$DEV -c focus_automatic_continuous=1
