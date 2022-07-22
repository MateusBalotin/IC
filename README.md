# Undergraduate Research Project

This repository contain 2 projects about Reinforcement Learning, more specifically of Q-Learning, which I did with my mentor Lucas Pedroso.

<h1> Cartpole </h1>

<p> This project is about a fairly common algorithm to understand DQN - Double Q-Networks, where the objective of the game <br>
is that the cart and pole doesn't move a lot. If the pole fall beyond a certain angle it's game over and the samme applies <br>
when the cart moves too far away</p>

<p> In the project I basically defined the fundamental things about Q-Learning and later on just called Neural Networs to train the model</p>

<h1> Freezing Lake</h1>


<p> This code is to show the basics of Q-Learning, especially Q-Table, without using any complicated means.</p>

<p> This game is basically one where you threw a frisbie to your dog in the middle of a ice lake and now you need to go get it,<br>
your dog is a lazy one. You can move to 9 places and the actions are left, right, up and down. <br>
There's 3 types of blocks, the blocks in which you can stand on, those that has a hole and it's game over if you go on them and <br>
the last one where the frisbie is.
</p>


<h1 align="center"> Q-Learning</h1>

<p> O aprendizado por reforço é uma área do Machine Learning que estuda como um agente em um ambiente, consegue maximizar uma certa recompensa. <br>

As aplicações do aprendizado por reforço são varidas, desde de jogos como Mario, Dota, até jogos de tabuleiro como Go, Xadrez <br> 
e self-driving cars, como os da Tesla.
</p>

<h2 align="center"> Markov Decision Process – MDP’S </h2>

<p> Um dos componentes principais do Aprendizado por Reforço são os processos de decisão de Markov:</p>

<ul> 
<li> Ambiente</li>
<li> Agente</li>
<li> Estado</li>
<li> Ação</li>
<li> Recompensa</li>
</ul>

<p>Vamos entender isso com um exemplo. Imagine que você está jogando xadrez com um amigo. <br>

Ambiente &rarr; é o conjunto de todas as regras e ações (jogadas) que você tem no xadrez <br>
Agente   &rarr; é quem interage com o ambiente e toma ações (você ou um algoritmo).
Estado   &rarr; é onde o agente se encontra para tomar ações. Por exemplo, quando o jogo de xadrez começa <br>
quem joga primeiro são quem está do lado das peças pretas. Logo, o estado em que o agente se encontra <br>
, a pessoa que vai jogar, é o estado inicial e ela pode fazer as ações permitidas.
Ação &rarr; é o ato de jogar. No nosso caso, é mover uma peça, seja um peão, cavalo.
Recompensa &rarr; é a forma de condicionar o agente a chegar em um certo objetivo. Imagina que você tem um pet <br>
e você quer que ele te dê a patinha toda vez que você der a mão para ele. Uma das formas de fazer isso, é que <br>
toda vez que ele da a patinha, você da um petisco para ele, ou seja, você esta condicionando para um certo objetivo <br>
que é dar a patinha toda vez que você estender a mão para ele. <br>
 A recompensa é exatamente isso, é o petisco que nós decidimos para o algoritmo para que ele alcance algum objetivo <br>
. Por exemplo, ganhar no jogo de xadrez ou dirigir de uma forma segura em um self driving car.
</p>


