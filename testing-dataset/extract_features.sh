#!/bin/bash

if [ -z "$1" ]
  then
    echo "Provide the absolute path to the DPC++ compiler folder as first argument"
	return
fi

DPCPP_CLANG=$1/clang++
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cmake -S $SCRIPT_DIR/passes -B $SCRIPT_DIR/passes/build
cmake --build $SCRIPT_DIR/passes/build -j

######################################################################################

alias sycl="$DPCPP_CLANG -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -fsycl-device-only -I$SCRIPT_DIR/sycl-bench/include/  -I$SCRIPT_DIR/sycl-bench/SYnergy/include -D __ENABLED_SYNERGY -D SYNERGY_KERNEL_PROFILING -D SYNERGY_CUDA_SUPPORT"
alias opt=opt-15

mkdir -p  $SCRIPT_DIR/bitcode
for file in $SCRIPT_DIR/sycl-bench/single-kernel/*.cpp; do
  name=`basename ${file%.*}`
  
  if [[ ! -f "$SCRIPT_DIR/bitcode/$name.bc" ]]; then
    sycl $file -o  $SCRIPT_DIR/bitcode/$name.bc
  fi 
done

mkdir -p $SCRIPT_DIR/features
mkdir -p $SCRIPT_DIR/features-count
mkdir -p $SCRIPT_DIR/features-normalized

# feature extraction from bitcode
for file in $SCRIPT_DIR/bitcode/*.bc; do
  name=`basename ${file%.*}`
  opt -load-pass-plugin $SCRIPT_DIR/passes/build/feature-pass/libfeature_pass.so  --passes="print<feature>" -disable-output $file 1>> "$SCRIPT_DIR/features/$name.temp" 2> /dev/null
done

for tempfile in $SCRIPT_DIR/features/*.temp; do
  name=`basename ${tempfile%.*}`

  features_count="$SCRIPT_DIR/features-count/${name}_features.csv"
  features_norm="$SCRIPT_DIR/features-normalized/${name}_features.csv"
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

type=("count" "normalized")

for t in "${type[@]}"; do
  file="$SCRIPT_DIR/features-$t/median_features.csv"
  features=$(cat $file | grep "_ZTS23MedianFilterBenchKernel")
  echo "kernel_name,mem_gl,int_add,int_bw,flt_mul,int_mul,flt_div,sp_fun,mem_loc,int_div,flt_add" > $file
  echo $features >> $file
done

for t in "${type[@]}"; do
  file="$SCRIPT_DIR/features-$t/scalar_prod_features.csv"
  features=$(cat $file | grep "_ZTS16ScalarProdKernelIfLb1EE")
  echo "kernel_name,mem_gl,int_add,int_bw,flt_mul,int_mul,flt_div,sp_fun,mem_loc,int_div,flt_add" > $file
  echo $features >> $file
done

rm -r $SCRIPT_DIR/features/