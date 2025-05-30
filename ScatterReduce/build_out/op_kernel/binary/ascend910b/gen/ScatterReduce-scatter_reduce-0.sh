#!/bin/bash
echo "[ascend910b] Generating ScatterReduce_b3292e28b5a330ade0fd6b28e66294aa ..."
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=1

while true; do
  case "$1" in
    --kernel-src=*)
      export BUILD_KERNEL_SRC=$(echo "$1" | cut -d"=" -f2-)
      shift
      ;;
    -*)
      shift
      ;;
    *)
      break
      ;;
  esac
done
res=$(opc $1 --main_func=scatter_reduce --input_param=/home/ma-user/work/ScatterReduce/build_out/op_kernel/binary/ascend910b/gen/ScatterReduce_b3292e28b5a330ade0fd6b28e66294aa_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/ScatterReduce_b3292e28b5a330ade0fd6b28e66294aa.json ; then
  echo "$2/ScatterReduce_b3292e28b5a330ade0fd6b28e66294aa.json not generated!"
  exit 1
fi

if ! test -f $2/ScatterReduce_b3292e28b5a330ade0fd6b28e66294aa.o ; then
  echo "$2/ScatterReduce_b3292e28b5a330ade0fd6b28e66294aa.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating ScatterReduce_b3292e28b5a330ade0fd6b28e66294aa Done"
