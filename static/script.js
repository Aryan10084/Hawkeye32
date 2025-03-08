document.addEventListener("keydown", function(event) {
    if (event.key === "ArrowUp") {
        document.getElementById("up").click();
    } else if (event.key === "ArrowDown") {
        document.getElementById("down").click();
    } else if (event.key === "ArrowLeft") {
        document.getElementById("left").click();
    } else if (event.key === "ArrowRight") {
        document.getElementById("right").click();
    } else if (event.key === " ") { // Spacebar for Stop
        document.getElementById("stop").click();
    }
});

// Dummy click functions for now (replace with actual control commands)
document.getElementById("up").onclick = () => console.log("Moving Up");
document.getElementById("down").onclick = () => console.log("Moving Down");
document.getElementById("left").onclick = () => console.log("Moving Left");
document.getElementById("right").onclick = () => console.log("Moving Right");
document.getElementById("stop").onclick = () => console.log("Stopping");
