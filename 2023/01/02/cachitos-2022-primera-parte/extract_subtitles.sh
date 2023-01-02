#!/bin/bash

root_directory=/home/jose/proyecto_cachitos
mkdir -p $root_directory
cd $root_directory

echo "First arg: $1"
mkdir -p video

cd video

ANNO=$1
echo $ANNO
suffix_video="_cachitos.mp4"
suffix_jpg_dir="_jpg"
suffix_txt_dir="_txt"

video_file=$ANNO$suffix_video
echo $video_file


if [ "$ANNO" == "2022" ] ;
then
    wget https://rtvehlsvodlote7.rtve.es/mediavodv2/resources/TE_SHIECRO/mp4/1/5/1672556504451.mp4
    mv 1672556504451.mp4 $video_file
fi


if [ "$ANNO" == "2021" ] ;
then
    wget https://lote5-vod-hls-geoblockurl.akamaized.net/resources/TE_GLUCA/mp4/4/0/1641020001504.mp4
    mv 1641020001504.mp4 $video_file
fi
 
if [ "$ANNO" == "2020" ] ;
then
    wget http://mediavod-lvlt.rtve.es/resources/TE_GLUCA/mp4/2/4/1609487028742.mp4
    mv 1609487028742.mp4 $video_file
fi

if [ "$ANNO" == "2019" ] ;
then
    wget https://rtvehlsvod2020a-fsly.vod-rtve.cross-media.es/resources/TE_GLUCA/mp4/0/9/1577860099590.mp4
    mv 1577860099590.mp4 $video_file
fi

# Pasar a jpg uno de cada 200 fotogramas

mplayer -vf framestep=200 -framedrop -nosound $video_file -speed 100 -vo jpeg:outdir=$ANNO$suffix_jpg_dir 
 
cd $ANNO$suffix_jpg_dir 
 
# Convertir a formato más pequño
find . -name '*.jpg' |  parallel -j 8 mogrify -resize 642x480 {}

# Seleccionar cacho dond están subtitulos
find . -name '*.jpg' |  parallel -j 8 convert {} -crop 460x50+90+295 +repage -compress none -depth 8 {}.subtitulo.tif

# Poner en negativo para que el ocr funcione mejor
find . -name '*.tif' |  parallel -j 8 convert {} -negate -fx '.8*r+.8*g+0*b' -compress none -depth 8 {}

# Pasar el ocr con idioma en español
find . -name '*.tif' |  parallel -j 8 tesseract -l spa {} {}

# mover a directorio texto
mkdir -p $root_directory/$ANNO$suffix_txt_dir

mv *.txt $root_directory/$ANNO$suffix_txt_dir

cd $root_directory
