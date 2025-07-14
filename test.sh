#!/bin/bash

echo "Testing the API with multiple image URLs..."

# List of image URLs
urls=(
  "https://ik.imagekit.io/2xkwa8s1i/img/npl_modified_images/Paintings-Images-new/WPTGSEGTP08S3/WPTGSEGTP08S3_LS_2.jpg?tr=w-640"
  "https://ik.imagekit.io/2xkwa8s1i/img/npl_modified_images/AUGAMDR32124/WSWB7860SAUGR3/WSWB7860SAUGR3_1.jpg?tr=w-640"
  "https://ik.imagekit.io/2xkwa8s1i/img/fitted-bedsheets/wfb/chekkers/3.jpg?tr=w-640"
  "https://ik.imagekit.io/2xkwa8s1i/helpdesk/image/file_1_IMG_1479.jpeg?tr=w-384"
  "https://ik.imagekit.io/2xkwa8s1i/img/fitted-bedsheets/wfb/chekkers/1.jpg?tr=w-640"
)

# Create output folder if missing
current_dir=$(pwd)
output_dir="$current_dir/out"
mkdir -p "$output_dir"

# Loop through each URL
for i in "${!urls[@]}"; do
  url="${urls[$i]}"
  original_file="$output_dir/original$((i+1)).jpg"
  output_file="$output_dir/sample$((i+1)).jpg"

  # Download original image if not already present
  if [ ! -f "$original_file" ]; then
    echo "Downloading original image $((i+1))..."
    curl -s "$url" --output "$original_file"
  else
    echo "Original image $((i+1)) already exists. Skipping download."
  fi

  # Always send image URL to API
  echo "Requesting prediction for image $((i+1))..."
  curl -s -X POST http://localhost:5000/predict \
       -H "Content-Type: application/json" \
       -d "{\"image_url\": \"$url\"}" \
       --output "$output_file"
done

echo "All requests completed."
