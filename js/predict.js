var results;

function getModelURL(){
    return 'https://bhanuchander210.github.io/naruto_eyes_classification/tfmodels/lenet/model.json';
}

function getImageSize()
{
    return 28;
}

async function main(){
    const imgEle = await document.getElementById('inimg');
    const model = await tf.loadLayersModel(getModelURL());

    // Getting the re-sized image
    var IMAGE_SIZE = getImageSize();
    var img = tf.browser.fromPixels(imgEle).resizeNearestNeighbor([IMAGE_SIZE, IMAGE_SIZE]).toFloat();
    console.log('Image')
    img.print()
    var normalized = img.mean(2).expandDims(2).div(255.0);
    normalized = normalized.expandDims();
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
        width: 700,
        height: 400,
        yaxis: {
            rangemode: 'nonnegative',
            range: [0, 100],
            showgrid: false,
            mirror: true,
            ticks: 'outside',
            showline: true,
            linewidth: 2
        },
        xaxis: {
            mirror: true,
            ticks: 'outside',
            showline: true,
            linewidth: 2
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

// Google Analytics
(function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
(i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
})(window,document,'script','//www.google-analytics.com/analytics.js','ga');

ga('create', 'UA-100320544-1', 'auto');
ga('send', 'pageview', {
  'page': 'https://bhanuchander210.github.io/naruto_eyes_classification',
  'title': 'Naruto_Eyes_classification'
});