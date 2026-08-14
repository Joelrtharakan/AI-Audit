function CheckOne(obj) {
    var grid = obj.parentNode.parentNode.parentNode;
    var row = obj.parentNode.parentNode;
    var rowindex = obj.parentNode.parentNode.rowIndex;
    var inputs = grid.getElementsByTagName("input");
    for (var i = 0; i < inputs.length; i++) {
        if (inputs[i].type == "checkbox") {
            if (obj.checked && inputs[i] != obj && inputs[i].checked) {
                inputs[i].checked = false;
                var count = 0;
                var rowid = 0;
                for (var ic in grid.rows) {
                    if (rowid < grid.rows.length) {
                        if (rowid != 0) {
                            if (rowid % 2 == 0) {
                                //Alternating Row Color
                                grid.rows[ic].style.backgroundColor = "#F0EEEE";
                            } else {

                                grid.rows[ic].style.backgroundColor = "white";
                            }
                        }
                        rowid = rowid + 1;
                    }
                }
                for (var i in grid.rows) {
                    if (count != 0) {

                        if (rowindex = (count + 1)) {
                            grid.rows[i].style.backgroundColor = "#A1DCF2";

                        }
                        count = count + 1;
                    }
                    $('#gvrightmenu').show();

                }
            }
        }
        var row = obj.parentNode.parentNode;
        if (obj.checked) {
            row.style.backgroundColor = "#A1DCF2";
            $('#gvrightmenu').show();
        } else {
            $('#gvrightmenu').hide();
            if (row.rowIndex % 2 == 0) {
                //Alternating Row Color
                row.style.backgroundColor = "#F0EEEE";
            } else {
                row.style.backgroundColor = "white";
            }
        }

    }
}


//function CheckOne(obj) {
//    var grid = obj.parentNode.parentNode.parentNode;
//    var row = obj.parentNode.parentNode;
//    var inputs = grid.getElementsByTagName("input");
//    for (var i = 0; i < inputs.length; i++) {
//        if (inputs[i].type == "checkbox") {
//            if (obj.checked && inputs[i] != obj && inputs[i].checked) {
//                inputs[i].checked = false;

//                $('#gvrightmenu').show();
//                // document.getElementById('Button3').click();
//            }
//        }
//    }
//    var row = obj.parentNode.parentNode;
//    if (obj.checked) {
//        $('#gvrightmenu').show();
//        // document.getElementById('Button3').click();
//    } else {
//        $('#gvrightmenu').hide();
//    }
//    // document.getElementById('btnHide').click();
//}
function MouseEvents(objRef, evt) {
    var checkbox = objRef.getElementsByTagName("input")[0];
    if (evt.type == "mouseover") {
        objRef.style.backgroundColor = "orange";
    }
    else {
        if (checkbox.checked) {
            objRef.style.backgroundColor = "aqua";
        }
        else if (evt.type == "mouseout") {
            if (objRef.rowIndex % 2 == 0) {
                //Alternating Row Color
                objRef.style.backgroundColor = "#F0EEEE";
            }
            else {
                objRef.style.backgroundColor = "white";
            }
        }
    }
}
function showGvActions() {
    $('#gvrightmenu').show();
}
function hideGvActions() {
    $('#gvrightmenu').hide();
}

function popitup(url) {
    var newwindow = window.open(url, 'name', 'height=800,width=1000');
    if (window.focus) {
        newwindow.focus();
    }
    return false;
}

function helpForm(helpId) {
    return popitup('../Help.aspx?helpId=' + helpId);
}


