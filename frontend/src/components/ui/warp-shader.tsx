"use client";

import { Warp } from "@paper-design/shaders-react";

export default function WarpShaderHero() {
    return (
        <div className="fixed inset-0 z-0">
            <Warp
                style={{ height: "100%", width: "100%" }}
                proportion={0.45}
                softness={1}
                distortion={0.25}
                swirl={0.8}
                swirlIterations={10}
                shape="checks"
                shapeScale={0.1}
                scale={1}
                rotation={0}
                speed={0.6}
                colors={[
                    "hsl(260, 80%, 15%)",
                    "hsl(280, 70%, 35%)",
                    "hsl(320, 60%, 25%)",
                    "hsl(200, 80%, 20%)",
                ]}
            />
        </div>
    );
}
