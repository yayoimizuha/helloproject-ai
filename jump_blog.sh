open "https://ameblo.jp/$(basename "$1" | cut -d '=' -f 2)/entry-$(basename "$1" | cut -d '=' -f 3 | cut -d '-' -f 1).html"
