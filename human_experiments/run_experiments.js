const jsPsych = initJsPsych();

const cifar_class_string =
  "<p>0: Airplane&nbsp;&nbsp;&nbsp;1: Automobile&nbsp;&nbsp;&nbsp;2: Bird&nbsp;&nbsp;&nbsp;3: Cat&nbsp;&nbsp;&nbsp;4: Deer</p><p>5: Dog&nbsp;&nbsp;&nbsp;6: Frog&nbsp;&nbsp;&nbsp;7: Horse&nbsp;&nbsp;&nbsp;8: Ship&nbsp;&nbsp;&nbsp;9: Truck</p>";
const fashion_mnist_class_string =
  "<p>0: T-shirt/top&nbsp;&nbsp;&nbsp;1: Trouser&nbsp;&nbsp;&nbsp;2: Pullover&nbsp;&nbsp;&nbsp;3: Dress&nbsp;&nbsp;&nbsp;4: Coat</p><p>5: Sandal&nbsp;&nbsp;&nbsp;6: Shirt&nbsp;&nbsp;&nbsp;7: Sneaker&nbsp;&nbsp;&nbsp;8: Bag&nbsp;&nbsp;&nbsp;9: Ankle boot</p>";
const initial_screen = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus:
    "<p>Welcome to this 9.60 project!</p><p><b>Instructions: (READ CAREFULLY)</b></p><p>We will show you a picture for a limited time and your job will be to classify the image.</p><p>The images will be rapid fire, so make sure you are ready to start.</p><p>There are 40 images, so this will take under 10 minutes.</p><p>(Press the key 'a' to start the experiment)</p>",
  choices: ["a"],
};

const finished_screen = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus: "<p>Done! Thanks for your help!</p>",
  choices: ["NO_KEYS"],
  on_start: function () {
    console.log(responses);
    console.log(responses.toString());
    const data = responses.toString(); // Data to be written to the file
    const fileName = "experiment_response.txt"; // Name of the file
    const blob = new Blob([data], { type: "text/plain" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = fileName;
    link.click();
  },
};

const before_cifar = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus:
    "<p>For this part, you will be selecting from the following ten classes.</p>" +
    cifar_class_string +
    "<p>(Press the letter 'b' to start the experiment)</p>",
  choices: ["b"],
};

const before_fashion = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus:
    "<p>For this part, you will be selecting from the following ten classes.</p>" +
    fashion_mnist_class_string +
    "<p>(Press the letter 'b' to start the experiment)</p>",
  choices: ["b"],
};

const short_break = {
  type: jsPsychHtmlKeyboardResponse,
  stimulus:
    "<p>Break</p><p>" +
    "</p><p>(Press the letter 'c' to continue the experiment)</p>",
  choices: ["c"],
};

const cifar10_image_urls = [
  "0_5.png",
  "1_4.png",
  "2_6.png",
  "3_7.png",
  "4_5.png",
  "5_8.png",
  "6_9.png",
  "7_6.png",
  "8_5.png",
  "9_5.png",
];

const fashion_mnist_image_urls = [
  "0_4.png",
  "1_4.png",
  "2_5.png",
  "3_3.png",
  "4_1.png",
  "5_4.png",
  "6_3.png",
  "7_0.png",
  "8_5.png",
  "9_1.png",
];

var responses = [];
var timeline = [];
timeline.push(initial_screen);
timeline.push(before_cifar);

// first 10 cifar for CNN
for (var i = 0; i < cifar10_image_urls.length; i++) {
  const cifar10_image_trial = {
    type: jsPsychImageKeyboardResponse,
    stimulus: "data/adversarial_examples/a/cnn/" + cifar10_image_urls[i],
    prompt:
      `<p>Question ${i + 1}</p>` +
      `
      <p>Here are the classes to choose from. Click the number on your keyboard of the class you want to select:</p>
      ` +
      cifar_class_string,
    choices: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    stimulus_height: 480,
    stimulus_width: 480,
    stimulus_duration: 1500,
    trial_duration: null,
    on_finish: function (data) {
      var response = data.response;
      responses.push(response);
    },
  };
  timeline.push(cifar10_image_trial);
}

timeline.push(short_break);

// first 10 cifar for Gabor
for (var i = 0; i < cifar10_image_urls.length; i++) {
  const cifar10_image_trial = {
    type: jsPsychImageKeyboardResponse,
    stimulus: "data/adversarial_examples/a/gabor/" + cifar10_image_urls[i],
    prompt:
      `<p>Question ${i + 1}</p>` +
      `
      <p>Here are the classes to choose from. Click the number on your keyboard of the class you want to select:</p>
      ` +
      cifar_class_string,
    choices: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    stimulus_height: 480,
    stimulus_width: 480,
    stimulus_duration: 1500,
    trial_duration: null,
    on_finish: function (data) {
      var response = data.response;
      responses.push(response);
    },
  };
  timeline.push(cifar10_image_trial);
}

timeline.push(before_fashion);

// first 10 fashion for CNN
for (var i = 0; i < fashion_mnist_image_urls.length; i++) {
  const fashion_mnist_image_trial = {
    type: jsPsychImageKeyboardResponse,
    stimulus: "data/adversarial_examples/b/cnn/" + fashion_mnist_image_urls[i],
    prompt:
      `<p>Question ${i + 1}</p>` +
      `
        <p>Here are the classes to choose from. Click the number on your keyboard of the class you want to select:</p>
        ` +
      fashion_mnist_class_string,
    choices: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    stimulus_height: 480,
    stimulus_width: 480,
    stimulus_duration: 1500,
    trial_duration: null,
    on_finish: function (data) {
      var response = data.response;
      responses.push(response);
    },
  };
  timeline.push(fashion_mnist_image_trial);
}

timeline.push(short_break);

// first 10 fashion for Gabor
for (var i = 0; i < fashion_mnist_image_urls.length; i++) {
  const fashion_mnist_image_trial = {
    type: jsPsychImageKeyboardResponse,
    stimulus:
      "data/adversarial_examples/b/gabor/" + fashion_mnist_image_urls[i],
    prompt:
      `<p>Question ${i + 1}</p>` +
      `
        <p>Here are the classes to choose from. Click the number on your keyboard of the class you want to select:</p>
        ` +
      fashion_mnist_class_string,
    choices: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    stimulus_height: 480,
    stimulus_width: 480,
    stimulus_duration: 1500,
    trial_duration: null,
    on_finish: function (data) {
      var response = data.response;
      responses.push(response);
    },
  };
  timeline.push(fashion_mnist_image_trial);
}

timeline.push(finished_screen);

console.log("starting experiment...");
jsPsych.run(timeline);
// console.log(responses);
