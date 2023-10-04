// function hideTextInput(val) {
//     if (document.getElementById("file_type") == "excel")
//     {
//         document.getElementById("delimeter").Visible=false; 
//     }
//     else
//     {
//         document.getElementById("delimeter").Visible=true; 
//     }
    
//   } 

function hideTextInput(selectedValue) {
    var delimiterLabel = document.querySelector('label[name="delimLabel"]');
    var delimiterInput = document.querySelector('input[name="delimiter"]');
    var uploadButton = document.querySelector('input[name="uploadButton"]');
    
    if (selectedValue === 'excel') {
        delimiterLabel.style.display = 'none';
        delimiterInput.style.display = 'none';
        // uploadButton.style.marginTop = '10px'; // Adjust the margin

    } else {
        delimiterLabel.style.display = 'inline-block'; // Display the label
        delimiterInput.style.display = 'inline-block'; // Display the input
        // uploadButton.style.marginTop = '0'; // Reset the margin
    }
}