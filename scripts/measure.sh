show_duration() {
    local ss=$1
    if test $ss -ge 86400; then
        local mm=$((ss / 60))
        local ss=$((ss % 60))

        local hh=$((mm / 60))
        local mm=$((mm % 60))

        local dd=$((hh / 24))
        local hh=$((hh % 24))

        echo "${dd}d${hh}h${mm}m${ss}s"
    elif test $ss -ge 3600; then
        local mm=$((ss / 60))
        local ss=$((ss % 60))

        local hh=$((mm / 60))
        local mm=$((mm % 60))

        echo "${hh}h${mm}m${ss}s"
    elif test $ss -ge 60; then
        local mm=$((ss / 60))
        local ss=$((ss % 60))

        echo "${mm}m${ss}s"
    else
        echo "${ss}s"
    fi
}

test_show_duration() {
    show_duration 100000
    show_duration 10000
    show_duration 100
    show_duration 10
}

measure() {
    if [ ! -z $SCRIPT_NAME ]; then
        local name=$SCRIPT_NAME
    else
        local name=$(basename $0)
    fi
    if [ ! -z $SCRIPT_DIR ]; then
        local log_dir=$SCRIPT_DIR
    else
        local log_dir=$(dirname $0)
    fi

    local begin=$(date +%s)
    echo "[begin] $name $ $@ at $begin"
    $@
    local end=$(date +%s)
    local duration=$((end - begin))
    local dur=$(show_duration $duration)
    echo "[done] $name $ $@ took ${dur}"
    # | tee -a $log_dir/profile.log
}
