<!doctype html>
<html lang="en">

<head>


  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">



  <title>Music Genre Recognition</title>



    <!--
    <link rel="stylesheet" href="{{ url_for('static',filename='style.css') }}">
   -->

      <script type="text/javascript" src="http://ajax.googleapis.com/ajax/libs/jquery/1.6.2/jquery.min.js"></script>


      <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">


      <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js" integrity="sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1" crossorigin="anonymous"></script>
      <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js" integrity="sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM" crossorigin="anonymous"></script>


      <style>
      .selector-for-some-widget {
        box-sizing: content-box;
      }
      </style>

</head>

<body>
        <h1>Music Genre Recognition using </h1><br>
		<h2><span>Convolutional Neural Networks and Long Short Term Memory</span></h2>
<!--
		<canvas id="canvas" width="280" height="280" style="border:8px solid; float: left; margin: 70px; margin-top:160px;  border-radius: 5px; cursor: crosshair;"></canvas>
		<div id="debug" style="margin:65px; margin-top:100px;">


	<input type="color" id="colors">
			 <input type="number" id="lineWidth" style="width:60px" value="0.2" step="0.1" min="0.1">






			<input type="button" id="clearButton" value="Clear" style=""  >
			<br/>


			<span style="color: #4DAF7C; font-weight: 400; font-family: 'Open Sans', Helvetica;	">Draw the Digit inside this Box!</span>
		</div>

-->

    <form method=post enctype=multipart/form-data>

      	<div style="margin-left:75px; margin-top:460px; float:left; position:absolute;">
          <input type=file name=file>
          </div>
          <div style="margin-left:75px; margin-top:560px; float:left; position:absolute;">
        <a href="#" class="myButton4" ><span style=" font-weight: 400; font-family: 'Open Sans', Helvetica;	">
          <input type=submit value=Upload onclick="loading();">
           </span></a>
           		</div>
    </form>


<!--
			<div style="margin-left:375px; margin-top:460px; float:left; position:absolute;">

			<a href="/" class="myButton3"><span style=" font-weight: 400; font-family: 'Open Sans', Helvetica;	"> Upload2 </span></a>
		</div>
-->






		<div style="margin-left:175px; margin-top:560px; float:left; position:absolute;">

			<a href="#" class="myButton" ><span style=" font-weight: 400; font-family: 'Open Sans', Helvetica;	"> Predict </span></a>
		</div>





				<div style="margin-left:275px; margin-top:560px; float:left; position:absolute;">

			<a href="/" class="myButton"><span style=" font-weight: 400; font-family: 'Open Sans', Helvetica;	"> Refresh </span></a>
		</div>



		<div >
			<h1 id="result" style="margin-right:20px; margin-top:300px;float:left;"><span> </span></h1>
		</div>


  		<script src='http://cdnjs.cloudflare.com/ajax/libs/jquery/2.1.3/jquery.min.js'></script>

	   <!--
     <script src="{{ url_for('static',filename='index.js') }}"></script>
     -->
	    <script type="text/javascript">


$body = $("body");
	   		$(".myButton").click(function(){
	   			var $SCRIPT_ROOT = {{ request.script_root|tojson|safe }};
	   		//	var canvasObj = document.getElementById("canvas");
	   		//	var img = canvasObj.toDataURL();
	   			var img="0";
	   			$.ajax({
	   				type: "POST",
	   				url: $SCRIPT_ROOT + "/predict/",
	   				data: img,
            beforeSend: function() {
               $body.addClass("loading");
           },
           complete: function() {
               $body.removeClass("loading");
           },
	   				success: function(data){
	   					$('#result').text(' Predicted Output: '+data);
	   				}
	   			});
	   		});


        $(".myButton4").click(function(){
         var $SCRIPT_ROOT = {{ request.script_root|tojson|safe }};
        // var canvasObj = document.getElementById("canvas");
        // var img = canvasObj.toDataURL();
         var img="0";
         $.ajax({
           type: "POST",

           beforeSend: function() {
              $body.addClass("loading2");
          },
          complete: function() {
              $body.removeClass("loading2");
          },
           success: function(data){
             $('#result').text(' Your music is loaded');
           }
         });
       });





	   		$(".myButton3").click(function(){
	   			var $SCRIPT_ROOT = {{ request.script_root|tojson|safe }};
	   		//	var canvasObj = document.getElementById("canvas");
	   		//	var img = canvasObj.toDataURL();
	   			var img="0";

	   			$.ajax({
	   				type: "POST",
	   				url: $SCRIPT_ROOT + "/upload/",
	   				data: img,
	   				success: function(data){
	   					$('#result').text(' ');
	   				}

	   			});

	   		});


	   </script>



<div id="loading" style="display:none;"><img src="/static/load2.gif" alt="" />Loading!</div>

<div class="modal"><!-- Place at bottom of page --></div>
<div class="modal2"><!-- Place at bottom of page --></div>
</body>
</html>
