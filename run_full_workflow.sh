#!/bin/bash
####################################################################################
# Emulator-based Calibration Full Workflow
#
# This script orchestrates the complete calibration workflow:
# 1. Generate 1000 parameter sets using Latin Hypercube Sampling
# 2. Extract forcing data from original LDASIN files (30-min resolution)
# 3. Run Noah-MP with each parameter set (using 30-min forcing)
# 4. Preprocess simulation outputs for emulator training
# 5. Train the LSTM emulator
# 6. Calibrate parameters using first-year data (201507-201607)
# 7. Validate calibration using full time range (201507-201707)
#
# Usage: bash run_full_workflow.sh [OPTIONS]
####################################################################################

set -e

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR="/home/petrichor/ymwang/snap/Emulator-based_calibration/calibration-BCI"
NOAHMP_BASE="/home/petrichor/ymwang/snap/Noah-mp"
FORCING_SOURCE="${NOAHMP_BASE}/data/Panama_single_point"
TBL_GENERATOR_DIR="${BASE_DIR}/noahmp/TBL_generator"
POINT_RUN_DIR="${BASE_DIR}/noahmp/point_run"

# Default settings
N_SAMPLES=1000
N_PARALLEL=4
SKIP_PARAM_GEN=false
SKIP_FORCING=false
SKIP_NOAHMP=false
SKIP_PREPROCESS=false
SKIP_TRAIN=false
SKIP_CALIBRATION=false

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${BASE_DIR}/logs/${TIMESTAMP}"

# =============================================================================
# Helper Functions
# =============================================================================
print_header() {
    echo ""
    echo "================================================================================="
    echo " $1"
    echo "================================================================================="
}

print_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

# =============================================================================
# Parse Arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-param-gen)
            SKIP_PARAM_GEN=true
            shift
            ;;
        --skip-forcing)
            SKIP_FORCING=true
            shift
            ;;
        --skip-noahmp)
            SKIP_NOAHMP=true
            shift
            ;;
        --skip-preprocess)
            SKIP_PREPROCESS=true
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --skip-calibration)
            SKIP_CALIBRATION=true
            shift
            ;;
        --parallel)
            N_PARALLEL="$2"
            shift 2
            ;;
        --samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-param-gen    Skip parameter generation"
            echo "  --skip-forcing      Skip forcing data extraction"
            echo "  --skip-noahmp       Skip Noah-MP runs"
            echo "  --skip-preprocess   Skip data preprocessing"
            echo "  --skip-train        Skip emulator training"
            echo "  --skip-calibration  Skip calibration"
            echo "  --parallel N        Parallel runs (default: 4)"
            echo "  --samples N         Number of samples (default: 1000)"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Initialize
# =============================================================================
print_header "EMULATOR-BASED CALIBRATION WORKFLOW"
print_info "Timestamp: ${TIMESTAMP}"
print_info "Base directory: ${BASE_DIR}"
print_info "Number of samples: ${N_SAMPLES}"
print_info "Parallel runs: ${N_PARALLEL}"

mkdir -p "${LOG_DIR}"
print_info "Log directory: ${LOG_DIR}"

cd "${BASE_DIR}"

# =============================================================================
# Step 1: Generate Parameter Sets
# =============================================================================
if [ "${SKIP_PARAM_GEN}" = false ]; then
    print_header "Step 1: Generate ${N_SAMPLES} Parameter Sets"
    
    cd "${TBL_GENERATOR_DIR}"
    sed -i "s/n_samples = .*/n_samples = ${N_SAMPLES}/" generate_samples.py
    python3 generate_samples.py 2>&1 | tee "${LOG_DIR}/01_param_generation.log"
    
    cp noahmp_${N_SAMPLES}samples_new.txt "${BASE_DIR}/data/raw/param/noahmp_param_sets.txt"
    print_success "Generated ${N_SAMPLES} parameter sets"
    cd "${BASE_DIR}"
else
    print_info "Skipping parameter generation"
fi

# =============================================================================
# Step 2: Extract Forcing Data (30-min resolution)
# =============================================================================
if [ "${SKIP_FORCING}" = false ]; then
    print_header "Step 2: Extract Forcing Data (30-min and daily)"
    
    # Extract 30-min resolution
    python3 extract_forcing_data.py \
        --forcing_dir "${FORCING_SOURCE}" \
        --output data/raw/forcing/forcing_panama_30min.nc \
        2>&1 | tee "${LOG_DIR}/02_forcing_extraction_30min.log"
    
    # Also extract daily resolution (for LSTM training if needed)
    python3 extract_forcing_data.py \
        --forcing_dir "${FORCING_SOURCE}" \
        --output data/raw/forcing/forcing_panama_daily.nc \
        --daily \
        2>&1 | tee "${LOG_DIR}/02_forcing_extraction_daily.log"
    
    print_success "Forcing data extracted (both 30-min and daily)"
else
    print_info "Skipping forcing data extraction"
fi

# =============================================================================
# Step 3: Run Noah-MP for All Parameter Sets
# =============================================================================
if [ "${SKIP_NOAHMP}" = false ]; then
    print_header "Step 3: Run Noah-MP Simulations (30-min forcing)"
    
    bash run_noahmp_batch.sh \
        --samples ${N_SAMPLES} \
        --parallel ${N_PARALLEL} \
        2>&1 | tee "${LOG_DIR}/03_noahmp_runs.log"
    
    print_success "Completed Noah-MP simulations"
else
    print_info "Skipping Noah-MP simulations"
fi

# =============================================================================
# Step 4: Data Preprocessing
# =============================================================================
if [ "${SKIP_PREPROCESS}" = false ]; then
    print_header "Step 4: Data Preprocessing"
    
    python3 01_data_preprocessing_forward_comprehensive.py \
        2>&1 | tee "${LOG_DIR}/04_preprocessing.log"
    
    print_success "Data preprocessing completed"
else
    print_info "Skipping data preprocessing"
fi

# =============================================================================
# Step 5: Train Emulator
# =============================================================================
if [ "${SKIP_TRAIN}" = false ]; then
    print_header "Step 5: Train LSTM Emulator"
    
    python3 02_train_forward_comprehensive.py \
        2>&1 | tee "${LOG_DIR}/05_training.log"
    
    MODEL_DIR=$(ls -td results_forward_comprehensive/*/ | head -1)
    print_success "Emulator trained: ${MODEL_DIR}"
    echo "${MODEL_DIR}" > "${LOG_DIR}/model_dir.txt"
else
    print_info "Skipping emulator training"
    MODEL_DIR=$(ls -td results_forward_comprehensive/*/ | head -1)
fi

# =============================================================================
# Step 6: Parameter Calibration
# =============================================================================
if [ "${SKIP_CALIBRATION}" = false ]; then
    print_header "Step 6: Parameter Calibration (First Year)"
    
    if [ -f "${LOG_DIR}/model_dir.txt" ]; then
        MODEL_DIR=$(cat "${LOG_DIR}/model_dir.txt")
    else
        MODEL_DIR=$(ls -td results_forward_comprehensive/*/ | head -1)
    fi
    
    python3 06_calibration_applying_emulator.py \
        --model_dir "${MODEL_DIR}" \
        --forcing data/raw/forcing/forcing_panama_daily.nc \
        --obs data/obs/Panama_BCI_obs_2015-07-30_2016-07-29.csv \
        --bounds value_bounds.csv \
        --num_calibration 10 \
        --max_iter 500 \
        --output calibration_results/${TIMESTAMP} \
        2>&1 | tee "${LOG_DIR}/06_calibration.log"
    
    print_success "Calibration completed"
fi

# =============================================================================
# Step 7: Validation
# =============================================================================
print_header "Step 7: Validation (Full Time Range)"

CALIBRATION_ID="${TIMESTAMP}/calibration_1"
bash 07_calibration_validation.sh "${CALIBRATION_ID}" \
    2>&1 | tee "${LOG_DIR}/07_validation.log"

print_success "Validation completed"

# =============================================================================
# Summary
# =============================================================================
print_header "WORKFLOW COMPLETED"
print_info "Logs: ${LOG_DIR}"
print_info "Model: ${MODEL_DIR}"
print_info "Calibration: calibration_results/${TIMESTAMP}"
print_success "Full workflow completed!"
