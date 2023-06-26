#!/bin/bash

CXX_COMPILER=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cxx_compiler=*)
            CXX_COMPILER="${1#*=}"
            shift
            ;;
        *)
            echo "Invalid argument: $1"
            return 1 2>/dev/null
            exit 1
            ;;
    esac
done

if [ -z "$CXX_COMPILER" ]
  then
    echo "Provide the absolute path to the DPC++ compiler as --cxx_compiler argument"
    return 1 2>/dev/null
    exit 1
fi

DPCPP_CLANG=$CXX_COMPILER
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cmake -S $SCRIPT_DIR/passes -B $SCRIPT_DIR/passes/build
cmake --build $SCRIPT_DIR/passes/build -j

######################################################################################

alias sycl="$DPCPP_CLANG -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -fsycl-device-only"
alias opt=opt-15

mkdir -p  $SCRIPT_DIR/micro/bitcode
for file in $SCRIPT_DIR/micro/*.cpp; do
  name=`basename ${file%.*}`
  
  if [[ ! -f "$SCRIPT_DIR/micro/bitcode/$name.bc" ]]; then
    echo "Generating bitcode for $name"
    sycl -Wno-unknown-cuda-version $file -o  $SCRIPT_DIR/micro/bitcode/$name.bc
  fi 
done

mkdir -p $SCRIPT_DIR/micro/features
mkdir -p $SCRIPT_DIR/micro/features-count
mkdir -p $SCRIPT_DIR/micro/features-normalized

# feature extraction from bitcode
for file in $SCRIPT_DIR/micro/bitcode/*.bc; do
  name=`basename ${file%.*}`
  opt -load-pass-plugin $SCRIPT_DIR/passes/build/feature-pass/libfeature_pass.so  --passes="print<feature>" -disable-output $file 1>> "$SCRIPT_DIR/micro/features/$name.temp" 2> /dev/null
done

for tempfile in $SCRIPT_DIR/micro/features/*.temp; do
  name=`basename ${tempfile%.*}`

  features_count="$SCRIPT_DIR/micro/features-count/${name}_features.csv"
  features_norm="$SCRIPT_DIR/micro/features-normalized/${name}_features.csv"
  echo "kernel_name,mem_gl,int_add,int_bw,flt_mul,int_mul,flt_div,sp_fun,mem_loc,int_div,flt_add" > $features_count
  echo "kernel_name,mem_gl,int_add,int_bw,flt_mul,int_mul,flt_div,sp_fun,mem_loc,int_div,flt_add" > $features_norm

  lines_number=$(sed -n "/IR-function/=" $tempfile)

  for n in $lines_number; do
    line_counters=""
    line_norm=""

    name="$(sed "${n}q;d" $tempfile | awk '{print $2}'),"
    line_counters+=$name
    line_norm+=$name

    counters_line_num=$(($n+6))
    counters=$(sed "${counters_line_num}q;d" $tempfile)
    let sum=0
    for num in $counters; do
      sum=$(( $sum + $num ))
    done
    if [ $sum -eq '0' ]; then
      continue
    fi
    line_counters+=$(echo $counters | sed 's/ /,/g')

    normalized_line_num=$(($n+8))
    normalized=$(sed "${normalized_line_num}q;d" $tempfile)
    line_norm+=$(echo $normalized | sed 's/ /,/g')

    echo $line_counters >> $features_count
    echo $line_norm >> $features_norm
  done
done

rm -r $SCRIPT_DIR/micro/features/