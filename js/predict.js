const modelURL = 'http://localhost:4000/tfmodels/lenet/model.json';
const IMAGE_SIZE = 28;
var results;

async function main(){
    const imgEle = await document.getElementById('inimg');
    const model = await tf.loadModel(modelURL);

    // Getting the re-sized image
    var img = tf.fromPixels(imgEle).resizeNearestNeighbor([IMAGE_SIZE, IMAGE_SIZE]).toFloat();
    var normalized = img.div(255.0).expandDims();

    normalized.print()
    const predicted = await model.predict(normalized).data()

    // get the model's prediction results
    results = Array.from(predicted)
    console.log('Predictions are : ', results)
    putChart (results)

}


initialCheck()
clearChart()

async function getMyImg(){
    return await document.getElementById('inimg');
}

async function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            $('#inimg').attr('src', e.target.result);
            $('#inimgShow').attr('src', e.target.result);
        };
        reader.readAsDataURL(input.files[0]);
        const imgEle = await getMyImg();
        imgEle.style.display = "block";
        clearChart();
        document.getElementById("predictBtn").disabled = false;
    }
}

$("#inputImg").change(function(){
        readURL(this);
});

function putChart(inData){

    inData[2] = inData[2] + inData[3]
    inData.pop()
    inData = inData.map(function(x){ return (x * 100).toFixed(2)})
    var data = [{
      x: ["Sharingan", "Byakugan", "Others"],
      y: inData,
      type: 'bar',
      text: inData.map(String),
      textposition: 'auto',
      marker:{
        color: ['rgba(222,45,38,0.8)', 'rgba(120,120,204,1)' , 'rgba(251,213,171,1)']
      }
    }];

    var layout = {
        title: 'Ocular Classification Result',
        width: 400,
        height: 300,
        plot_bgcolor:"#c2b999",
        paper_bgcolor:"#c2b999",
        yaxis: {
        rangemode: 'nonnegative',
        range: [0, 100]
        },
        font: {
            size: 9,
        },
        margin: {
            l: 50,
            r: 50,
            b: 50,
            t: 80,
            pad: 4
        }
    }

    Plotly.newPlot('plot', data, layout);
    document.getElementById('plot').style.display = 'block';
}


function initialCheck(){

    if (window.File && window.FileReader && window.FileList && window.Blob) {
        console.log("success! All the File APIs are supported.")
    } else {
        alert('The File APIs are not fully supported in this browser.');
    }

}

function clearChart(){
    putChart([0, 0, 0, 0])
    console.log("Chart cleared.")
}