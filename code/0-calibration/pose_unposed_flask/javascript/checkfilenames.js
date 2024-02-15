<form id="upload-form" method="post" enctype="multipart/form-data">
    <input id="file-input" type="file" name="files[]" multiple>
    <input type="submit" value="Upload">
</form>

<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
<script>
$(document).ready(function() {
    $('#upload-form').on('submit', function(e) {
        var files = $('#file-input')[0].files;
        for (var i = 0; i < files.length; i++) {
            var filename = files[i].name;
            if (!isValid(filename)) {
                alert('Invalid file: ' + filename);
                e.preventDefault();  // stop the form from submitting
                return;
            }
        }
    });

    function isValid(filename) {
        // Add your validation logic here. For example, you might check the file extension:
        return filename.endsWith('.jpg') || filename.endsWith('.png');
    }
});
</script>