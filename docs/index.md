<style>
.home-container {
    max-width: 800px;
    margin: 2rem auto;
    text-align: center;
    padding: 2rem;
}

.profile-pic {
    border-radius: 50%;
    width: 200px;
    height: 200px;
    margin: 0 auto 2rem;
    display: block;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.home-title {
    font-size: 2.5rem;
    margin-bottom: 1rem;
    font-weight: 600;
}

.home-subtitle {
    font-size: 1.25rem;
    margin-bottom: 2rem;
    opacity: 0.8;
}

.social-links {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-top: 2rem;
}

.social-links a {
    font-size: 2rem;
    transition: transform 0.2s;
    opacity: 0.7;
}

.social-links a:hover {
    transform: scale(1.1);
    opacity: 1;
}

.intro-section {
    margin-top: 3rem;
    text-align: left;
    line-height: 1.8;
}

.intro-section h2 {
    margin-top: 2rem;
    margin-bottom: 1rem;
}
</style>

<div class="home-container">
    <img src="./img/photo.jpg" alt="Murtaza Nazir" class="profile-pic"/>

    <h1 class="home-title">Murtaza Nazir</h1>
    <p class="home-subtitle">Building AI one tiny piece at a time.</p>

    <div class="social-links">
        <a href="https://github.com/themurtazanazir/" target="_blank" title="GitHub">
            :fontawesome-brands-github:
        </a>
        <a href="https://twitter.com/murtazanazir/" target="_blank" title="Twitter">
            :fontawesome-brands-twitter:
        </a>
    </div>

    <div class="intro-section">
        <h2>Welcome</h2>
        <p>
            This is my personal space where I explore and document my journey through artificial intelligence,
            machine learning, and deep learning. Here you'll find notes on linear algebra, neural networks,
            and various AI concepts that I'm learning and implementing.
        </p>

        <h2>Topics</h2>
        <p>
            📐 <strong>Linear Algebra</strong> - Foundations of machine learning mathematics<br/>
            🧠 <strong>Neural Networks</strong> - From perceptrons to transformers
        </p>
    </div>
</div>

