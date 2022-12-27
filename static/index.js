function getFileName() {
    let file = document.getElementById('filePath');
    let filePath = file.value;
    let allowed = /(\.jpg|\.jpeg|\.png|\.svg|\.jfif)$/i;
    if (file.files.length) {
        if (!allowed.exec(filePath)) {
            alert('Невірний формат файлу');
            file.value = '';
            return false;
        } else {
            for (var i = 0; i <= file.files.length - 1; i++) {
                document.getElementById('showName').innerHTML =
                    document.getElementById('showName').innerHTML +
                    file.files.item(i).name;
            }
        }
    }
}


