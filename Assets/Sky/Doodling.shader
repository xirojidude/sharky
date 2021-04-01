
Shader "Skybox/Doodling"
{
    Properties
    {
        _MainTex ("tex2D", 2D) = "white" {}
        _MainTex2 ("tex2D", 2D) = "white" {}
        _SunDir ("Sun Dir", Vector) = (-.11,.07,0.99,0) 
        _XYZPos ("XYZ Offset", Vector) = (0, 15, -.25 ,0) 
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            uniform sampler2D _MainTex; 
            uniform sampler2D _MainTex2; 
            float4 _SunDir,_XYZPos;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f {
                float4 uv : TEXCOORD0;         //posWorld
                float4 worldPos : TEXCOORD1;
                float4 vertex : SV_POSITION;   //pos
                float4 screenPos : TEXCOORD2;
            };

            v2f vert (appdata v) {
                appdata v2;
                v2.vertex = v.vertex; //mul(v.vertex ,float4x4(-1,0.,0.,0.,  0.,1.,0.,0.,  0.,0.,1.,0.,  0.,0.,0.,1.));
                v2f o = (v2f)0;
                o.uv = mul(unity_ObjectToWorld, v2.vertex);
                o.vertex = UnityObjectToClipPos( v.vertex); // * float4(1.0,1.,1.0,1.) ); // squish skybox sphere
                o.worldPos = mul(unity_ObjectToWorld, v2.vertex);
                o.screenPos  = ComputeScreenPos(o.vertex);
                //o.screenPos.z = -(mul(UNITY_MATRIX_MV, vertex).z * _ProjectionParams.w);     // old calculation, as I used the depth buffer comparision for min max ray march. 

                return o;
            }


// @lsdlive

// Doodling session for live-coding or sketching new ideas.
// Thanks to iq, mercury, lj, shane, shau, aiekick, balkhan
// & all shadertoyers.
// Greets to all the shader showdown paris gang.


float2x2 r2d(float a) {
    float c = cos(a), s = sin(a);
    return float2x2(c, s, -s, c);
}

void amod(inout float2 p, float m) {
    float a = ((atan2( p.y,p.x) - m*.5)% m) - m*.5;
    p = float2(cos(a), sin(a)) * length(p);
}

float rep(float p, float m) {
    return ((p - m*.5)% m) - m*.5;
}

float3 rep(float3 p, float m) {
    return ((p - m*.5)% m) - m*.5;
}

float cmin(float a, float b, float k) {
    return min(min(a, b), (a - k + b) * sqrt(.5));
}

float stmin(float a, float b, float k, float n) {
    float s = k / n;
    float u = b - k;
    return min(min(a, b), .5 * (u + a + abs((((u - a + s)% 2. * s)) - s)));
}

float length8(float2 p) {
    float2 q = p*p*p*p*p*p*p*p;
    return pow(q.x + q.y, 1. / 8.);
}

float torus82(float3 p, float2 d) {
    float2 q = float2(length(p.xz) - d.x, p.y);
    return length8(q) - d.y;
}

float path(float t) {
    return cos(t*.1)*2.;
}

float g = 0.;
float de(float3 p) {

    p.x -= path(p.z);
    
    float3 q = p;
    q.x += sin(q.z*.2)*4.;
    q.y += cos(q.z*.3)*4.;
    q += _Time.y*2.;
    q.yz += sin(_Time.y*.2)*4.;
    q = rep(q, 1.);
    float s1 = length(q) - .01 + sin(_Time.y*30.)*.004;

    p.z = rep(p.z, 3.);

    float d = torus82(p.xzy, float2(1., .1));
    float pl = p.y + 2.4 + p.y*tex2D(_MainTex2, p.xz*.1).r*1.;
    float pl2 = p.y + .7;
    d = min(d, pl-d*.9);
    d = cmin(d, pl2, .1);

    amod(p.xy, 6.28 / 3.);
    p.x = abs(p.x) - 1.;
    float cyl = length(p.xy) - .05;
    d = stmin(d, cyl, .1, 4.);
   
    d = min(d, s1);

    g += .015 / (.01 + d*d);
    return d;
}


         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;
                float2 screenUV = v.screenPos.xy / v.screenPos.w;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+ _XYZPos;                                             // ray origin

//    float2 uv = (fragCoord - .5*iResolution.xy) / iResolution.y;

    float dt = _Time.y * 8.3;

    ro = float3(0, 0, -3. + dt);
    float3 ta = float3(0, 0, dt);
    ro.x += path(ro.z);
    ta.x += path(ta.z);

    float3 fwd = normalize(ta - ro);
    float3 left = cross(float3(0, 1, 0), fwd);
    float3 up = cross(fwd, left);
//    float3 rd = normalize(fwd + uv.x*left + uv.y*up);

//    rd.xy *= r2d(sin(-ro.x / 3.14)*.4);

    float3 p;
    float t = 0., ri;
    for (float i = 0.; i < 1.; i += .01) {
        ri = i;
        p = ro + rd*t;
        float d = de(p);
        if (d < .001) break;
        t += d*.3;
    }

    float3 bg = float3(.2, .1, .2);
    float3 col = bg;
//    col = lerp(float3(.4, .52, .6)*1.5, bg,  uv.x*uv.y*uv.y+ri);
    col += g*.01;
    col.b += sin(p.z*.1)*.1;
    col = lerp(col, bg, 1. - exp(-.01*t*t));
    
    fragColor = float4(col, 1.);


                return fragColor;
            }

            ENDCG
        }
    }
}

